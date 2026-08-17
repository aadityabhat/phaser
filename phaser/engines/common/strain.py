"""Strain-gradient distortion correction (autodiff-native port of `distort.py`/`gradcalc.py`
from the CuPy ptycho pipeline this branch is being translated from -- see
`docs/theory/strain_gradient_distortion.pdf` for the underlying math).

Two halves, matching the paper:

1. `strain_perturbation` -- called from `engines/gradient/run.py`'s `run_model` when
   'distortion' is an active var. Implements O_sg(eps) = O - sum_ij eps_ij (x_j-x_j0)
   dO/dx_i as a delta added to the gathered object transmission window, evaluated at
   eps=0. This leaves the forward model exactly unchanged (eps=0 identically, not just
   at a point that later moves), so autodiff yields d(loss)/d(eps) at that point -- the
   strain-gradient sensitivity -- with no need for a hand-derived adjoint.

2. `StrainDistortionSolver` -- a `GradientSolver` with params={'distortion'} that
   consumes the accumulated (npos, 4) strain gradient once per iteration and produces
   {'object', 'positions'} updates via a Fourier-Laplacian-inversion Poisson solve
   (ported near-verbatim from distort.py, CPU/SciPy -- no JAX/torch equivalent for the
   scattered/gridded interpolation steps).

Note on sign: `tree.grad(..., sign=-1)` (used by `run_group` to extract iter_grads)
already returns the *descent direction*, not the raw dLoss/deps -- so
`StrainDistortionSolver.update` scales `grad['distortion']` directly, with no extra
negation (unlike a plain `jax.grad` result, which would need one).
"""
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass
from phaser.utils.num import brake, get_array_module, to_real_dtype, to_numpy
from phaser.hooks.solver import GradientSolver, GradientSolverArgs

if t.TYPE_CHECKING:
    from phaser.state import ObjectState, ReconsState


def strain_perturbation(
    obj: 'ObjectState',
    group_scan: NDArray[numpy.floating],
    eps: NDArray[numpy.floating],
    cutout_shape: t.Tuple[int, ...],
) -> NDArray[numpy.complexfloating]:
    """Delta to add to a `get_view_at_pos`-gathered object window, implementing
    O_sg(eps) = O - sum_ij eps_ij (x_j - x_j0) dO/dx_i.

    `eps`: (batch, 4) = [eps_xx, eps_xy, eps_yx, eps_yy], indexed as (derivative
    direction i, offset direction j) with i,j in {x, y} -- matching gradcalc.py's
    d_d channel order (row/x-offset, row/y-offset, col/x-offset, col/y-offset).
    `group_scan`: (batch, 2), (y, x) -- same positions `obj.sampling.get_view_at_pos`
    was already called with to produce the object window being perturbed.

    Returned in the object's *natural* (unshifted) pixel layout -- callers using the
    ifft2shift-preshifted `group_obj` convention (see run_model) must ifft2shift this
    delta too before adding, so it lines up pixel-for-pixel.

    Object gradients are per unit length (divided by pixel sampling), so `eps` is a
    dimensionless strain, matching the physical convention.
    """
    xp = get_array_module(obj.data)
    sampling = obj.sampling.sampling  # (s_y, s_x)

    (ny, nx) = cutout_shape[-2:]

    (dy_full, dx_full) = xp.gradient(obj.data, axis=(-2, -1))
    dy_full = dy_full / sampling[0]
    dx_full = dx_full / sampling[1]

    group_dy = obj.sampling.get_view_at_pos(dy_full, group_scan, cutout_shape)
    group_dx = obj.sampling.get_view_at_pos(dx_full, group_scan, cutout_shape)

    # subpixel shift from the (rounded) window center towards the true position,
    # in length units, (y, x) -- same quantity used to Fourier-shift the probe.
    subpx = obj.sampling.get_subpx_shifts(group_scan, cutout_shape)
    dtype = to_real_dtype(subpx.dtype)

    y_local = (xp.arange(ny, dtype=dtype) - (ny - 1) / 2.) * sampling[0]
    x_local = (xp.arange(nx, dtype=dtype) - (nx - 1) / 2.) * sampling[1]

    # (x_j - x_j0) per pixel per position, shape (batch, 1, ny/nx, 1/nx) for broadcast
    # against (batch, n_slices, ny, nx) gradient windows.
    off_y = y_local[None, None, :, None] - subpx[:, 0][:, None, None, None]
    off_x = x_local[None, None, None, :] - subpx[:, 1][:, None, None, None]

    eps_xx = eps[:, 0][:, None, None, None]
    eps_xy = eps[:, 1][:, None, None, None]
    eps_yx = eps[:, 2][:, None, None, None]
    eps_yy = eps[:, 3][:, None, None, None]

    delta = (
        -(eps_xx * off_x + eps_xy * off_y) * group_dx
        - (eps_yx * off_x + eps_yy * off_y) * group_dy
    )
    # ObjectSampling's host-side sampling/subpx-shift arithmetic runs at float64
    # regardless of the reconstruction's working dtype, so `delta` may have been
    # promoted above group_obj's precision (e.g. complex64 -> complex128 under a
    # float32 plan). Cast back to match group_obj exactly: at eps=0 the value is
    # exactly zero regardless of dtype, so this doesn't affect the "forward model
    # unchanged at eps=0" guarantee, and letting a higher-precision delta leak into
    # group_obj breaks jax.lax.scan's multislice loop, which requires a fixed carry
    # dtype throughout (observed as a scan carry dtype-mismatch crash on real data).
    return delta.astype(obj.data.dtype)


def _scattered_linear_zero_fill(points: NDArray, values: NDArray, query: NDArray) -> NDArray:
    """Linear inside the convex hull of `points`, zero outside."""
    from scipy.interpolate import LinearNDInterpolator
    result = LinearNDInterpolator(points, values, fill_value=numpy.nan)(query)
    return numpy.nan_to_num(result, nan=0.0)


def _gridded_linear_nearest_extrap(grid_axes: t.Tuple[NDArray, NDArray], values: NDArray, query: NDArray) -> NDArray:
    """Linear inside a regular grid; boundary value held constant outside (clamp-then-linear == nearest extrapolation)."""
    from scipy.interpolate import RegularGridInterpolator
    lo = numpy.array([axis.min() for axis in grid_axes])
    hi = numpy.array([axis.max() for axis in grid_axes])
    clamped = numpy.clip(query, lo, hi)
    return RegularGridInterpolator(grid_axes, values, method="linear")(clamped)


def _scattered_linear_nearest(points: NDArray, values: NDArray, query: NDArray) -> NDArray:
    """Linear inside the convex hull of `points`, nearest-neighbor outside."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    lin = LinearNDInterpolator(points, values, fill_value=numpy.nan)(query)
    near = NearestNDInterpolator(points, values)(query)
    nan_mask = numpy.isnan(lin.real) if numpy.iscomplexobj(lin) else numpy.isnan(lin)
    return numpy.where(nan_mask, near, lin)


def solve_distortion(
    scan_idx: NDArray[numpy.floating],
    obj_data: NDArray[numpy.complexfloating],
    delta_eps: NDArray[numpy.floating],
) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.floating]]:
    """Convert a per-position strain *update* (already scaled to a step size and in
    descent direction -- NOT the raw loss gradient) into object and position updates.

    `scan_idx`: (npos, 2) scan positions in the object's own pixel-index space (y, x),
    i.e. `(scan - sampling.corner) / sampling.sampling`. NOT a windowed cutout.
    `obj_data`: (n_slices, ny, nx) complex object, on the same pixel grid as `scan_idx`.
    `delta_eps`: (npos, 4) = [eps_xx, eps_xy, eps_yx, eps_yy] step, same channel order
    as `strain_perturbation`'s `eps`.

    Returns `(new_obj, new_scan_idx)` (new_scan_idx still in pixel-index space).

    Implements straingradient.pdf section 3: scatter the per-position strain onto the
    object's pixel grid (linear inside the convex hull, zero outside -- since the
    resulting displacement field must stay continuous), Fourier-invert the discrete
    Laplacian to get a smooth global displacement field consistent with all the local
    strain samples, then use that field to shift positions and forward-warp the object.
    CPU/SciPy only -- no JAX/torch equivalent for the scattered/gridded interpolation.
    """
    scan_idx = to_numpy(scan_idx).astype(numpy.float64)
    obj_data = to_numpy(obj_data)
    delta_eps = to_numpy(delta_eps).astype(numpy.float64)

    (n_slices, ny, nx) = obj_data.shape
    (ry, rx) = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx), indexing="ij")
    grid_yx = numpy.stack([ry.ravel(), rx.ravel()], axis=1)

    # discrete central-difference derivative operators, as convolution kernels
    kernel_dy = numpy.zeros((ny, nx))
    kernel_dy[1, 0] = 0.5
    kernel_dy[ny - 1, 0] = -0.5
    kernel_dx = numpy.zeros((ny, nx))
    kernel_dx[0, 1] = 0.5
    kernel_dx[0, nx - 1] = -0.5
    Dy = numpy.fft.fft2(kernel_dy)
    Dx = numpy.fft.fft2(kernel_dx)
    L = Dy**2 + Dx**2  # discrete Laplacian's frequency response

    # frequencies the Laplacian can't sample (DC + Nyquist-like points): these are
    # unrealistically high/low frequencies for a real displacement field, zero them.
    zero_mask = numpy.abs(L) < 1e-10
    L_safe = numpy.where(zero_mask, 1.0, L)

    re = _scattered_linear_zero_fill(scan_idx, delta_eps, grid_yx).reshape(ny, nx, 4)

    # eq. 5: u_i = ifft[ (1/L) * sum_j fft[D_j] * fft[eps_ij] ], i,j in {x, y}
    Ux_f = (Dx * numpy.fft.fft2(re[:, :, 0]) + Dy * numpy.fft.fft2(re[:, :, 1])) / L_safe
    Uy_f = (Dx * numpy.fft.fft2(re[:, :, 2]) + Dy * numpy.fft.fft2(re[:, :, 3])) / L_safe
    Ux_f = numpy.where(zero_mask, 0, Ux_f)
    Uy_f = numpy.where(zero_mask, 0, Uy_f)

    ux = numpy.real(numpy.fft.ifft2(Ux_f))  # col (x) displacement, pixel units
    uy = numpy.real(numpy.fft.ifft2(Uy_f))  # row (y) displacement, pixel units

    # position update: x_new = x_old + u_interp(x_old)
    y_axis = numpy.arange(ny, dtype=numpy.float64)
    x_axis = numpy.arange(nx, dtype=numpy.float64)
    disp_at_pos = _gridded_linear_nearest_extrap((y_axis, x_axis), numpy.stack([uy, ux], axis=2), scan_idx)
    new_scan_idx = (scan_idx + disp_at_pos).astype(numpy.float32)

    # object update: O_new(x + u) = O_old(x) -> interpolate O_old at (x + u), evaluate at x
    warped_yx = numpy.stack([(ry + uy).ravel(), (rx + ux).ravel()], axis=1)
    obj_flat = obj_data.transpose(1, 2, 0).reshape(ny * nx, n_slices)
    new_re = _scattered_linear_nearest(warped_yx, obj_flat.real, grid_yx)
    new_im = _scattered_linear_nearest(warped_yx, obj_flat.imag, grid_yx)
    new_obj = (new_re + 1j * new_im).reshape(ny, nx, n_slices).transpose(2, 0, 1).astype(obj_data.dtype)

    return new_obj, new_scan_idx


class StrainDistortionSolverProps(Dataclass):
    step_size: float = 1e-2
    """Fraction of the strain gradient to convert into a displacement-producing update."""
    max_step_size: t.Optional[float] = None
    """Maximum per-position strain-update magnitude, soft-clipped (see `phaser.utils.num.brake`)
    before the Poisson solve."""


class StrainDistortionSolver(GradientSolver[None]):
    """Consumes the per-position strain gradient (see `strain_perturbation`) and produces
    coupled `{'object', 'positions'}` updates via `solve_distortion`. Must be used as a
    per-iteration solver keyed on `{'distortion'}` (see `_PER_ITER_VARS` in
    `engines/gradient/run.py`) since the Poisson solve needs the whole scan's strain
    samples at once, not a single group's.
    """
    name = 'strain_distortion'

    def __init__(self, args: GradientSolverArgs, props: StrainDistortionSolverProps):
        self.params = frozenset(args['params'])
        if self.params != frozenset({'distortion'}):
            raise ValueError(
                f"'strain_distortion' must be registered for exactly {{'distortion'}}, "
                f"got {set(self.params)!r}"
            )
        self.step_size = props.step_size
        self.max_step_size = props.max_step_size

    def init_state(self, sim: 'ReconsState') -> None:
        return None

    def update_for_iter(self, sim: 'ReconsState', state: None, niter: int) -> None:
        return state

    def update(
        self, sim: 'ReconsState', state: None, grad: t.Dict[str, NDArray[numpy.floating]], loss: float,
    ) -> t.Tuple[t.Dict[str, t.Any], None]:
        xp = get_array_module(grad['distortion'])

        # grad['distortion'] is already the descent direction (tree.grad's sign=-1),
        # not the raw dLoss/deps -- scale directly, don't negate again.
        delta_eps = self.step_size * grad['distortion']
        if self.max_step_size is not None:
            delta_eps = brake(delta_eps, self.max_step_size)

        corner = sim.object.sampling.corner
        sampling = sim.object.sampling.sampling
        scan_idx = (to_numpy(sim.scan) - corner) / sampling

        (new_obj, new_scan_idx) = solve_distortion(scan_idx, to_numpy(sim.object.data), to_numpy(delta_eps))

        new_scan = new_scan_idx * sampling + corner
        object_delta = xp.asarray(new_obj - to_numpy(sim.object.data), dtype=sim.object.data.dtype)
        positions_delta = xp.asarray(new_scan - to_numpy(sim.scan), dtype=sim.scan.dtype)

        return ({'object': object_delta, 'positions': positions_delta}, state)
