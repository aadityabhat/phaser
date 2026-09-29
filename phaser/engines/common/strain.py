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
   {'object', 'positions'} updates via a Fourier-Laplacian-inversion Poisson solve.

`solve_distortion` (2026-09-21 rewrite, matching CuPy's own `distort.py` rewrite of the same
period): scattered strain samples are smoothed onto the object grid via Nadaraya-Watson kernel
regression (`_scattered_kernel_regression`) rather than a linear-inside-hull/zero-outside fit --
the old fit's hard discontinuity at the scan-hull boundary rang through the Poisson solve below (a
global, periodic-FFT operation) and damaged the object near the edge of the scanned region. The
object update is a backward/"pull" warp with fixed-point refinement (`_pull_warp_object`) rather
than a forward/"push" scatter-and-retriangulate, which broke down near sharp object features even
for the small per-iteration displacements actually seen in production. Both fixes ported from
CuPy's `recon_12slices/ptycho/distort.py`; see that module's docstring for the original
derivation/validation. Runs entirely on the active `xp` backend (JAX/CuPy/Torch/NumPy) via
`phaser.utils.num.scatter_add`/`phaser.utils.image.map_coordinates` -- no CPU round-trip, unlike
the version this replaced (which had no cross-backend equivalent for the old
LinearNDInterpolator-based scatter).

Note on sign: `tree.grad(..., sign=-1)` (used by `run_group` to extract iter_grads)
already returns the *descent direction*, not the raw dLoss/deps -- so
`StrainDistortionSolver.update` scales `grad['distortion']` directly, with no extra
negation (unlike a plain `jax.grad` result, which would need one).
"""
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass
from phaser.utils.num import brake, invavg, get_array_module, to_real_dtype, at, scatter_add, fft2, ifft2
from phaser.utils.image import map_coordinates
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


def _max_kth_nn_dist(points: NDArray[numpy.floating], k: int, batch_size: int = 4096) -> t.Any:
    """Distance to each point's k-th nearest OTHER point, maxed over all points -- the data-driven
    bandwidth `_scattered_kernel_regression` uses (ported from CuPy's `distort.py`
    `_max_kth_nn_dist`, itself a GPU-portable replacement for `scipy.spatial.cKDTree`). Brute-force
    over batches of query points (O(n^2) total, trivial at the ~1e4-1e5 scan-position counts this
    is used for); uses `xp.sort` rather than CuPy's `xp.partition` for broader backend support
    (JAX/torch don't universally expose partition) -- O(n log n) instead of O(n), immaterial here.
    Runs eagerly (this whole module is called outside any `@jit`), so plain Python looping over
    `n`/`batch_size` is safe even under JAX.
    """
    xp = get_array_module(points)
    n = points.shape[0]
    sq_norms = xp.sum(points ** 2, axis=1)
    max_kth_sq = xp.asarray(0.0, dtype=points.dtype)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        chunk = points[start:stop]
        d2 = sq_norms[start:stop][:, None] + sq_norms[None, :] - 2.0 * (chunk @ points.T)
        d2 = xp.maximum(d2, 0.0)
        kth_sq = xp.sort(d2, axis=1)[:, k]
        max_kth_sq = xp.maximum(max_kth_sq, xp.max(kth_sq))
    return xp.sqrt(max_kth_sq)


def _fft_gaussian_blur(field: NDArray[numpy.floating], sigma: t.Any, ny: int, nx: int) -> NDArray[numpy.floating]:
    """Circular (periodic) Gaussian blur via the analytic FT of a Gaussian, over `field`'s last two
    axes -- reuses the same FFT-multiply pattern `solve_distortion`'s own Poisson solve uses."""
    xp = get_array_module(field)
    real_dtype = field.dtype
    ky = xp.fft.fftfreq(ny).astype(real_dtype)[:, None]
    kx = xp.fft.fftfreq(nx).astype(real_dtype)[None, :]
    gaussian_ft = xp.exp(-2 * (xp.pi * sigma) ** 2 * (ky ** 2 + kx ** 2))
    return xp.real(ifft2(fft2(field, shift=False) * gaussian_ft, shift=False)).astype(real_dtype)


def _scattered_kernel_regression(
    points: NDArray[numpy.floating],
    values: NDArray[numpy.floating],
    grid_shape: t.Tuple[int, int],
    min_neighbors: int = 6,
) -> NDArray[numpy.floating]:
    """Nadaraya-Watson kernel regression of scattered (points, values) onto a regular 0-indexed
    (y, x) grid, via FFT splat-blur-divide:

        f(x) = sum_i K(x - x_i) v_i / sum_i K(x - x_i),   K = isotropic Gaussian

    Ported from CuPy's `distort.py` `_scattered_kernel_regression` (see that module's docstring
    for the full derivation/validation) -- replaces a linear-inside-convex-hull/zero-outside fit,
    whose hard boundary discontinuity rang through the Poisson solve below and damaged the object
    near the edge of the scanned region. This decays smoothly to zero away from the scan instead.

    Bandwidth is entirely data-driven (no per-dataset tuning constant): sigma is the smallest value
    such that every point has at least `min_neighbors` other points within one bandwidth of it
    (`_max_kth_nn_dist`), so every estimate inside the scanned region averages over at least
    `min_neighbors` independent samples. The same blurred splat denominator doubles as a smooth
    local-density estimate, giving a decay envelope with no extra computation and no hard cutoff
    (hence nothing for the Poisson solve to ring on).

    `points` (n,2), (y, x) in the same 0-indexed pixel coordinates as the target grid. `values` is
    (n,) or (n,k); returns `grid_shape` (or `grid_shape + (k,)`).
    """
    xp = get_array_module(points, values)
    dtype = to_real_dtype(values.dtype)
    points = points.astype(dtype)
    trailing = values.shape[1:]
    values2 = xp.reshape(values, (values.shape[0], -1)).astype(dtype)
    nchan = values2.shape[1]
    (ny, nx) = grid_shape

    sigma = xp.maximum(_max_kth_nn_dist(points, min_neighbors), xp.asarray(1e-6, dtype=dtype))

    y0 = xp.floor(points[:, 0]).astype(numpy.int64)
    x0 = xp.floor(points[:, 1]).astype(numpy.int64)
    fy = points[:, 0] - y0.astype(dtype)
    fx = points[:, 1] - x0.astype(dtype)
    (y0m, y1m) = (y0 % ny, (y0 + 1) % ny)
    (x0m, x1m) = (x0 % nx, (x0 + 1) % nx)

    # bilinear splat (adjoint of bilinear gather), onto the periodic grid the FFT blur below
    # already treats this grid as having.
    corners = (
        (y0m, x0m, (1 - fy) * (1 - fx)),
        (y0m, x1m, (1 - fy) * fx),
        (y1m, x0m, fy * (1 - fx)),
        (y1m, x1m, fy * fx),
    )

    den = xp.zeros((ny, nx), dtype=dtype)
    for (yi, xi, w) in corners:
        den = den + scatter_add((ny, nx), (yi, xi), w, dtype=dtype)

    num_channels = []
    for c in range(nchan):
        chan_grid = xp.zeros((ny, nx), dtype=dtype)
        for (yi, xi, w) in corners:
            chan_grid = chan_grid + scatter_add((ny, nx), (yi, xi), w * values2[:, c], dtype=dtype)
        num_channels.append(chan_grid)
    num = xp.stack(num_channels, axis=0)  # (nchan, ny, nx)

    num_b = _fft_gaussian_blur(num, sigma, ny, nx)
    den_b = _fft_gaussian_blur(den, sigma, ny, nx)

    # tau: a small fraction of `min_neighbors` points' worth of the densest sampled location's
    # own weight -- ties the only remaining constant back to min_neighbors, not a separate tunable.
    tau = xp.maximum(xp.max(den_b), xp.asarray(1e-300, dtype=dtype)) / (2.0 * min_neighbors)
    den_safe = den_b + tau
    envelope = den_b / den_safe
    result = (num_b / den_safe[None, :, :]) * envelope[None, :, :]
    result = xp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    result = xp.moveaxis(result, 0, -1)  # (ny, nx, nchan)
    return result.reshape((ny, nx) + trailing) if trailing else result[:, :, 0]


def _sample_field_at_positions(
    field: NDArray[numpy.floating], query_yx: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    """field: (ny,nx) or (ny,nx,k) real array on the object's own 0-indexed pixel grid.
    query_yx: (n,2) query coordinates in that same grid. Linearly interpolated, with
    out-of-bounds queries clamped to the nearest valid grid point. Returns (n,) or (n,k)."""
    xp = get_array_module(field, query_yx)
    coords = xp.stack([query_yx[:, 0], query_yx[:, 1]])
    if field.ndim == 2:
        return map_coordinates(field, coords, order=1, mode='nearest')
    trailing = field.shape[2:]
    field2 = xp.reshape(field, (field.shape[0], field.shape[1], -1))
    out = xp.stack(
        [map_coordinates(field2[:, :, k], coords, order=1, mode='nearest') for k in range(field2.shape[2])],
        axis=-1,
    )
    return out.reshape((query_yx.shape[0],) + trailing)


_PULL_WARP_REFINE_ITERS = 2  # see _pull_warp_object's docstring


def _pull_warp_object(
    obj_data: NDArray[numpy.complexfloating], uy: NDArray[numpy.floating], ux: NDArray[numpy.floating],
    refine_iters: int = _PULL_WARP_REFINE_ITERS,
) -> NDArray[numpy.complexfloating]:
    """Backward/pull warp: obj_new(x) = obj(y(x)), where y solves y + u(y) = x (the true inverse of
    "the value at y moved to y+u(y)"), via ordinary regular-grid interpolation -- replaces a
    forward/"push" scatter-and-retriangulate warp, which is only well-posed where the warp stays
    locally injective and produced real artifacts near sharp object features even for the small,
    non-folding per-iteration displacements actually seen in production (ported from CuPy's
    `distort.py` `_pull_warp_object`; see that module's docstring for the full argument).

    y is found by fixed-point iteration, y <- x - u(y), starting from y0 = x - u(x) (exact only as
    u->0, accurate to O(|grad(u)|^2) -- fine at production's sub-pixel displacements but not for a
    single larger correction). Each refinement step needs u evaluated at the current estimate y --
    itself just a regular-grid interpolation, no triangulation, so no folding/holes/degenerate-
    triangle failure mode is possible regardless of `refine_iters`.

    obj_data: (n_slices, ny, nx) complex. uy, ux: (ny, nx) real displacement field, same grid.
    """
    xp = get_array_module(obj_data, uy, ux)
    (n_slices, ny, nx) = obj_data.shape
    dtype = uy.dtype
    (ry0, rx0) = xp.meshgrid(xp.arange(ny, dtype=dtype), xp.arange(nx, dtype=dtype), indexing='ij')
    (yy, xx) = (ry0 - uy, rx0 - ux)
    for _ in range(refine_iters):
        coords_y = xp.stack([yy.ravel(), xx.ravel()])
        u_at_y_y = map_coordinates(uy, coords_y, order=1, mode='nearest').reshape(ny, nx)
        u_at_y_x = map_coordinates(ux, coords_y, order=1, mode='nearest').reshape(ny, nx)
        (yy, xx) = (ry0 - u_at_y_y, rx0 - u_at_y_x)

    coords = xp.stack([yy.ravel(), xx.ravel()])
    slices = []
    for k in range(n_slices):
        real_k = map_coordinates(xp.real(obj_data[k]), coords, order=1, mode='nearest').reshape(ny, nx)
        imag_k = map_coordinates(xp.imag(obj_data[k]), coords, order=1, mode='nearest').reshape(ny, nx)
        slices.append(real_k + 1j * imag_k)
    return xp.stack(slices, axis=0).astype(obj_data.dtype)


def solve_distortion(
    scan_idx: NDArray[numpy.floating],
    obj_data: NDArray[numpy.complexfloating],
    delta_eps: NDArray[numpy.floating],
    min_neighbors: int = 6,
    refine_iters: int = _PULL_WARP_REFINE_ITERS,
) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.floating]]:
    """Convert a per-position strain *update* (already scaled to a step size and in descent
    direction -- NOT the raw loss gradient) into object and position updates.

    `scan_idx`: (npos, 2) scan positions in the object's own pixel-index space (y, x), i.e.
    `(scan - sampling.corner) / sampling.sampling`. NOT a windowed cutout.
    `obj_data`: (n_slices, ny, nx) complex object, on the same pixel grid as `scan_idx`.
    `delta_eps`: (npos, 4) = [eps_xx, eps_xy, eps_yx, eps_yy] step, same channel order as
    `strain_perturbation`'s `eps`.

    Returns `(new_obj, new_scan_idx)` (new_scan_idx still in pixel-index space).

    Implements straingradient.pdf section 3: scatter the per-position strain onto the object's
    pixel grid (`_scattered_kernel_regression`), Fourier-invert the discrete Laplacian to get a
    smooth global displacement field consistent with all the local strain samples, then use that
    field to shift positions and backward-warp the object (`_pull_warp_object`). Runs entirely on
    the active `xp` backend -- see this module's docstring.
    """
    xp = get_array_module(scan_idx, obj_data, delta_eps)
    dtype = to_real_dtype(obj_data.dtype)
    scan_idx = scan_idx.astype(dtype)
    delta_eps = delta_eps.astype(dtype)

    (n_slices, ny, nx) = obj_data.shape

    # discrete central-difference derivative operators, as convolution kernels
    kernel_dy = at(xp.zeros((ny, nx), dtype=dtype), (1, 0)).set(0.5)
    kernel_dy = at(kernel_dy, (ny - 1, 0)).set(-0.5)
    kernel_dx = at(xp.zeros((ny, nx), dtype=dtype), (0, 1)).set(0.5)
    kernel_dx = at(kernel_dx, (0, nx - 1)).set(-0.5)
    Dy = fft2(kernel_dy, shift=False)
    Dx = fft2(kernel_dx, shift=False)
    L = Dy ** 2 + Dx ** 2  # discrete Laplacian's frequency response

    # frequencies the Laplacian can't sample (DC + Nyquist-like points): unrealistically
    # high/low frequencies for a real displacement field -- zero them.
    zero_mask = xp.abs(L) < 1e-10
    L_safe = xp.where(zero_mask, 1.0, L)

    # scattered strain samples -> smooth grid, [eps_xx, eps_xy, eps_yx, eps_yy] channel order
    # (matching strain_perturbation's `eps`).
    re = _scattered_kernel_regression(scan_idx, delta_eps, (ny, nx), min_neighbors=min_neighbors)

    # eq. 5: u_i = ifft[ (1/L) * sum_j fft[D_j] * fft[eps_ij] ], i,j in {x, y}
    Ux_f = (Dx * fft2(re[:, :, 0], shift=False) + Dy * fft2(re[:, :, 1], shift=False)) / L_safe
    Uy_f = (Dx * fft2(re[:, :, 2], shift=False) + Dy * fft2(re[:, :, 3], shift=False)) / L_safe
    Ux_f = xp.where(zero_mask, 0, Ux_f)
    Uy_f = xp.where(zero_mask, 0, Uy_f)

    ux = xp.real(ifft2(Ux_f, shift=False))  # col (x) displacement, pixel units
    uy = xp.real(ifft2(Uy_f, shift=False))  # row (y) displacement, pixel units

    # position update: x_new = x_old + u_interp(x_old)
    disp_at_pos = _sample_field_at_positions(xp.stack([uy, ux], axis=-1), scan_idx)
    new_scan_idx = (scan_idx + disp_at_pos).astype(dtype)

    new_obj = _pull_warp_object(obj_data, uy, ux, refine_iters=refine_iters).astype(obj_data.dtype)

    return new_obj, new_scan_idx


class StrainDistortionSolverProps(Dataclass):
    step_size: float = 1e-2
    """Target update magnitude: the raw strain gradient is rescaled by `step_size / invavg(grad)`
    (see `phaser.utils.num.invavg`) before braking, so the applied step tracks this scale
    regardless of the gradient's own magnitude -- a direct port of the CuPy reference
    implementation's `df = dm / invavg(d['d'])` normalization (`python/ptycho/optimize.py`)."""
    max_step_size: t.Optional[float] = None
    """Maximum per-position strain-update magnitude, soft-clipped (see `phaser.utils.num.brake`)
    before the Poisson solve."""
    min_neighbors: int = 6
    """Data-driven kernel-regression bandwidth target -- see `_scattered_kernel_regression`."""
    decay_half_life: t.Optional[float] = 1000.0
    """`step_size` is scaled by `0.5 ** (total_iter / decay_half_life)` each iteration
    (`total_iter` = `sim.iter.total_iter`, resume-aware), before the `invavg` normalization
    above. `None` disables the decay (flat `step_size`). Direct port of CuPy's
    `0.5 ** (i / 1000)` factor on `df_raw` (`python/ptycho/optimize.py`)."""
    warmup_freeze_gate_iter: t.Optional[int] = 250
    """Once `sim.iter.total_iter` exceeds this, the per-iteration (decayed, invavg-normalized)
    step is replaced by a running average of its own raw value, accumulated over the next
    `warmup_freeze_iters` iterations, then frozen at that average for every iteration after --
    plain fixed-step descent from there on. `None` disables freezing (the decayed,
    per-iteration-normalized step is used every iteration, matching a fresh reconstruction's
    very first `warmup_freeze_gate_iter` iterations regardless of this setting). Direct port of
    CuPy's `SCAN_STEP_WARMUP_ITERS`/`i > 250` gate on `df` (`python/ptycho/optimize.py`); see that
    module's docstring for the full motivation (the per-iteration signal was found too small/noisy
    near an already-converged checkpoint to show clean accumulation without this -- memory:
    12slice-divergence-ablation-20260914). Not resume-safe by design, matching CuPy: the running
    average itself always restarts from scratch (`init_state`), even if `total_iter` resumes
    partway through what would have been its own warmup window."""
    warmup_freeze_iters: int = 100
    """See `warmup_freeze_gate_iter`."""


class _StrainStepState(t.NamedTuple):
    warmup_sum: float = 0.0
    warmup_count: int = 0
    frozen: t.Optional[float] = None


class StrainDistortionSolver(GradientSolver[_StrainStepState]):
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
        self.min_neighbors = props.min_neighbors
        self.decay_half_life = props.decay_half_life
        self.warmup_freeze_gate_iter = props.warmup_freeze_gate_iter
        self.warmup_freeze_iters = props.warmup_freeze_iters

    def init_state(self, sim: 'ReconsState') -> _StrainStepState:
        return _StrainStepState()

    def update_for_iter(self, sim: 'ReconsState', state: _StrainStepState, niter: int) -> _StrainStepState:
        return state

    def update(
        self, sim: 'ReconsState', state: _StrainStepState, grad: t.Dict[str, NDArray[numpy.floating]], loss: float,
    ) -> t.Tuple[t.Dict[str, t.Any], _StrainStepState]:
        # grad['distortion'] is already the descent direction (tree.grad's sign=-1),
        # not the raw dLoss/deps -- scale directly, don't negate again.
        raw_grad = grad['distortion']
        xp = get_array_module(raw_grad)
        total_iter = int(sim.iter.total_iter)

        decay = 1.0 if self.decay_half_life is None else 0.5 ** (total_iter / self.decay_half_life)
        # xp.maximum(..., eps) guards a still-zero gradient (e.g. distortion's very first
        # active iteration), where invavg's own sum(|.|)/sum(|.|^2) would otherwise be 0/0.
        raw_step_scale = float(decay * self.step_size / xp.maximum(invavg(raw_grad), 1e-30))

        if total_iter <= 1:
            # No prior gradient history yet on a fresh reconstruction's very first iteration --
            # skip the distortion update entirely rather than normalize off a single, possibly
            # degenerate sample (matches CuPy's `always_apply_dsdf or i > 1` gate; every real
            # caller there passes `always_apply_dsdf=False` for a fresh run and starts resumes
            # past iteration 1, so this is the gate's whole effective behavior).
            step_scale = 0.0
        elif self.warmup_freeze_gate_iter is not None and total_iter > self.warmup_freeze_gate_iter:
            if state.frozen is not None:
                step_scale = state.frozen
            else:
                warmup_sum = state.warmup_sum + raw_step_scale
                warmup_count = state.warmup_count + 1
                step_scale = warmup_sum / warmup_count
                frozen = step_scale if warmup_count >= self.warmup_freeze_iters else None
                state = _StrainStepState(warmup_sum, warmup_count, frozen)
        else:
            step_scale = raw_step_scale

        delta_eps = step_scale * raw_grad
        if self.max_step_size is not None:
            delta_eps = brake(delta_eps, self.max_step_size)

        corner = sim.object.sampling.corner
        sampling = sim.object.sampling.sampling
        scan_idx = (sim.scan - corner) / sampling

        (new_obj, new_scan_idx) = solve_distortion(
            scan_idx, sim.object.data, delta_eps, min_neighbors=self.min_neighbors,
        )

        new_scan = new_scan_idx * sampling + corner
        object_delta = (new_obj - sim.object.data).astype(sim.object.data.dtype)
        positions_delta = (new_scan - sim.scan).astype(sim.scan.dtype)

        return ({'object': object_delta, 'positions': positions_delta}, state)
