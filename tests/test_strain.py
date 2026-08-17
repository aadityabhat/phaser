"""Tests for strain-gradient distortion correction (`phaser.engines.common.strain`).

See `strain.py`'s module docstring / `docs/theory/strain_gradient_distortion.pdf` for
the underlying math: perturbing the sampled object by O_sg(eps) = O - sum_ij eps_ij
(x_j-x_j0) dO/dx_i and evaluating at eps=0 should leave the forward model exactly
unchanged, while autodiff's d(loss)/d(eps) equals the local strain-gradient
sensitivity -- checked here against a direct finite-difference derivative of the same
loss function. Also checks the sign convention against `tree.grad(..., sign=-1)`
(already the descent direction, unlike a plain `jax.grad` result) and that
`StrainDistortionSolver`'s Poisson-solved update measurably decreases loss.
"""
import numpy
import pytest

from phaser.utils.num import get_backend_module

pytest.importorskip("jax")
jnp = get_backend_module('jax')  # loading via phaser's backend loader enables jax_enable_x64
import jax  # noqa: E402

from phaser.utils.num import Sampling
from phaser.utils.object import ObjectSampling
from phaser.state import ObjectState, ProbeState, ReconsState, IterState
from phaser.plan import AmplitudeNoisePlan
from phaser.engines.common.noise_models import AmplitudeNoiseModel
from phaser.engines.gradient.run import run_model, SolverStates
from phaser.engines.common.strain import StrainDistortionSolver, StrainDistortionSolverProps, strain_perturbation
from phaser.engines.common.simulation import make_propagators
import phaser.utils.tree as tree


def _make_toy_problem(seed: int = 0):
    rng = numpy.random.default_rng(seed)

    (ny, nx) = (24, 24)
    (by, bx) = (10, 10)
    npos = 6

    obj_sampling = ObjectSampling((ny, nx), sampling=(1.0, 1.0))
    obj_phase = rng.normal(0, 0.3, (1, ny, nx))
    obj_amp = 1.0 - 0.1 * rng.random((1, ny, nx))
    obj_data = jnp.array((obj_amp * numpy.exp(1j * obj_phase)).astype(numpy.complex128))
    obj_state = ObjectState(obj_sampling, obj_data, jnp.array([1.0]))

    probe_sampling = Sampling((by, bx), sampling=(1.0, 1.0))
    probe_data = jnp.array((rng.normal(0, 1, (1, by, bx)) + 1j * rng.normal(0, 1, (1, by, bx))).astype(numpy.complex128))
    probe_state = ProbeState(probe_sampling, probe_data)

    # keep positions comfortably inside the object, away from the boundary
    scan = jnp.array(rng.uniform(-4, 4, (npos, 2)))

    state = ReconsState(
        iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state,
        scan=scan, tilt=None, progress={},
    )

    patterns = jnp.array(rng.uniform(0.1, 1.0, (npos, by, bx)))
    mask = jnp.ones((by, bx))
    noise_model = AmplitudeNoiseModel(None, AmplitudeNoisePlan())
    solver_states = SolverStates(noise_model_state=None, group_solver_states=[], regularizer_states=[], group_constraint_states=[])
    # group's last axis is the batch size; scan's leading shape is 1-D, so a single index row
    group = jnp.arange(npos)[None, :]

    return (state, patterns, mask, noise_model, solver_states, group, npos)


def _loss(state, patterns, mask, noise_model, solver_states, group, vars_dict):
    (loss, _aux) = run_model(
        vars_dict, state, group=group, props=None,
        group_patterns=patterns, pattern_mask=mask,
        noise_model=noise_model, regularizers=(), solver_states=solver_states,
        xp=jnp, dtype=numpy.float64, jit_unroll_slices=False,
    )
    return loss


def test_strain_perturbation_unchanged_at_zero():
    """O_sg(eps=0) must leave the forward model (and thus the loss) exactly unchanged,
    whether or not 'distortion' is present as a key at all."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    eps0 = jnp.zeros((npos, 4))
    loss_with_eps = _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps0})
    loss_without = _loss(state, patterns, mask, noise_model, solver_states, group, {})

    assert loss_with_eps == loss_without


def test_strain_gradient_matches_finite_difference():
    """autodiff's d(loss)/d(eps) must match a direct central-difference derivative."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    eps0 = jnp.zeros((npos, 4))
    autodiff_grad = numpy.asarray(jax.grad(loss_fn)(eps0))

    h = 1e-4
    fd_grad = numpy.zeros((npos, 4))
    for pos_i in range(npos):
        for ch in range(4):
            ep = numpy.array(eps0)
            em = numpy.array(eps0)
            ep[pos_i, ch] += h
            em[pos_i, ch] -= h
            fd_grad[pos_i, ch] = (float(loss_fn(jnp.array(ep))) - float(loss_fn(jnp.array(em)))) / (2 * h)

    numpy.testing.assert_allclose(autodiff_grad, fd_grad, atol=1e-4, rtol=1e-3)


def test_strain_gradient_sign_matches_tree_grad_descent_direction():
    """tree.grad(..., sign=-1), what run_group actually uses, must return exactly
    -raw_grad (the descent direction) for real-valued eps -- StrainDistortionSolver
    relies on this and must NOT re-negate."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    eps0 = jnp.zeros((npos, 4))
    raw_grad = numpy.asarray(jax.grad(loss_fn)(eps0))
    descent_grad = numpy.asarray(tree.grad(loss_fn, xp=jnp, sign=-1)(eps0))

    numpy.testing.assert_allclose(descent_grad, -raw_grad, atol=1e-8)


def test_strain_solver_decreases_loss_end_to_end():
    """StrainDistortionSolver's Poisson-solved {object, positions} update should decrease
    the loss -- exercises the Fourier-Laplacian inversion and interpolation, not just the
    eps-gradient extraction covered by the other tests."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    eps0 = jnp.zeros((npos, 4))
    loss0 = float(loss_fn(eps0))
    descent_grad = tree.grad(loss_fn, xp=jnp, sign=-1)(eps0)

    solver = StrainDistortionSolver({'plan': None, 'params': frozenset({'distortion'})}, StrainDistortionSolverProps(step_size=0.5))
    (update, _) = solver.update(state, None, {'distortion': descent_grad}, loss0)

    assert numpy.all(numpy.isfinite(numpy.asarray(update['object'])))
    assert numpy.all(numpy.isfinite(numpy.asarray(update['positions'])))

    new_object = ObjectState(state.object.sampling, state.object.data + update['object'], state.object.thicknesses)
    new_state = ReconsState(
        iter=state.iter, wavelength=state.wavelength, probe=state.probe, object=new_object,
        scan=state.scan + update['positions'], tilt=None, progress=state.progress,
    )

    loss_after = float(_loss(new_state, patterns, mask, noise_model, solver_states, group, {}))
    assert loss_after < loss0


def test_strain_perturbation_preserves_object_dtype():
    """Regression test: ObjectSampling's host-side sampling/subpx-shift math runs at
    float64 regardless of the reconstruction's working dtype, which previously let a
    float32 (complex64) object's perturbation delta silently promote to complex128.
    Adding that into group_obj inside a multislice reconstruction crashes
    jax.lax.scan (fixed carry dtype required across the slice loop) -- caught live on
    a real dataset. Exercise the actual multislice/jax.lax.scan path (n_slices > 1),
    not just the single-slice shortcut the other tests use."""
    rng = numpy.random.default_rng(1)

    (ny, nx) = (24, 24)
    (by, bx) = (10, 10)
    npos = 5
    n_slices = 3

    obj_sampling = ObjectSampling((ny, nx), sampling=(1.0, 1.0))
    obj_phase = rng.normal(0, 0.3, (n_slices, ny, nx)).astype(numpy.float32)
    obj_amp = (1.0 - 0.1 * rng.random((n_slices, ny, nx))).astype(numpy.float32)
    obj_data = jnp.array(obj_amp * numpy.exp(1j * obj_phase), dtype=jnp.complex64)
    obj_state = ObjectState(obj_sampling, obj_data, jnp.array([50.0, 50.0, 50.0], dtype=jnp.float32))

    probe_sampling = Sampling((by, bx), sampling=(1.0, 1.0))
    probe_data = jnp.array(
        (rng.normal(0, 1, (1, by, bx)) + 1j * rng.normal(0, 1, (1, by, bx))).astype(numpy.complex64)
    )
    probe_state = ProbeState(probe_sampling, probe_data)
    scan = jnp.array(rng.uniform(-4, 4, (npos, 2)), dtype=jnp.float32)

    state = ReconsState(
        iter=IterState.empty(), wavelength=0.025, probe=probe_state, object=obj_state,
        scan=scan, tilt=None, progress={},
    )

    eps = jnp.zeros((npos, 4), dtype=jnp.float32)
    delta = strain_perturbation(obj_state, scan, eps, (by, bx))
    assert delta.dtype == obj_data.dtype, f"expected {obj_data.dtype}, got {delta.dtype}"

    patterns = jnp.array(rng.uniform(0.1, 1.0, (npos, by, bx)), dtype=jnp.float32)
    mask = jnp.ones((by, bx), dtype=jnp.float32)
    noise_model = AmplitudeNoiseModel(None, AmplitudeNoisePlan())
    solver_states = SolverStates(noise_model_state=None, group_solver_states=[], regularizer_states=[], group_constraint_states=[])
    group = jnp.arange(npos)[None, :]
    props = make_propagators(state, bwlim_frac=None)

    (loss, _aux) = run_model(
        {'distortion': eps}, state, group=group, props=props,
        group_patterns=patterns, pattern_mask=mask,
        noise_model=noise_model, regularizers=(), solver_states=solver_states,
        xp=jnp, dtype=numpy.float32, jit_unroll_slices=False,
    )
    assert numpy.isfinite(float(loss))
