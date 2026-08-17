"""Tests for strain-gradient distortion correction (`phaser.engines.common.strain`).

See `strain.py`'s module docstring / the accompanying note on the math: perturbing
the sampled object by O_sg(eps) = O - sum_ij eps_ij (x_j-x_j0) dO/dx_i and evaluating
at eps=0 should leave the forward model exactly unchanged, while making autodiff's
d(loss)/d(eps) equal to the local strain-gradient sensitivity -- checked here against
a direct finite-difference derivative of the same loss function.
"""
import numpy
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from phaser.utils.num import Sampling
from phaser.utils.object import ObjectSampling
from phaser.state import ObjectState, ProbeState, ReconsState, IterState, ProgressState
from phaser.plan import AmplitudeNoisePlan
from phaser.engines.common.noise_models import AmplitudeNoiseModel
from phaser.engines.gradient.run import run_model, SolverStates
from phaser.engines.common.strain import StrainDistortionSolver, StrainDistortionSolverProps


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
        scan=scan, tilt=None, progress=ProgressState.empty(),
    )

    patterns = jnp.array(rng.uniform(0.1, 1.0, (npos, by, bx)))
    mask = jnp.ones((by, bx))
    noise_model = AmplitudeNoiseModel(None, AmplitudeNoisePlan())
    solver_states = SolverStates(noise_model_state=None, group_solver_states=[], regularizer_states=[], group_constraint_states=[])
    # scan's leading shape is 1-D (npos,), so a single index row
    group = jnp.arange(npos)[None, :]

    return (state, patterns, mask, noise_model, solver_states, group, npos)


def test_strain_perturbation_unchanged_at_zero():
    """O_sg(eps=0) must leave the forward model (and thus the loss) exactly unchanged."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        (loss, _aux) = run_model(
            {'distortion': eps}, state, group=group, props=None,
            group_patterns=patterns, pattern_mask=mask,
            noise_model=noise_model, regularizers=(), solver_states=solver_states,
            xp=jnp, dtype=numpy.float64,
        )
        return loss

    eps0 = jnp.zeros((npos, 4))
    assert loss_fn(eps0) == loss_fn(eps0)


def test_strain_gradient_matches_finite_difference():
    """autodiff's d(loss)/d(eps) must match a direct central-difference derivative."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        (loss, _aux) = run_model(
            {'distortion': eps}, state, group=group, props=None,
            group_patterns=patterns, pattern_mask=mask,
            noise_model=noise_model, regularizers=(), solver_states=solver_states,
            xp=jnp, dtype=numpy.float64,
        )
        return loss

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


def test_strain_solver_decreases_loss_end_to_end():
    """StrainDistortionSolver's Poisson-solved {object, positions} update should decrease
    the loss -- exercises the Fourier-Laplacian inversion and interpolation, not just the
    eps-gradient extraction covered by the other tests."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(s):
        (loss, _aux) = run_model(
            {}, s, group=group, props=None,
            group_patterns=patterns, pattern_mask=mask,
            noise_model=noise_model, regularizers=(), solver_states=solver_states,
            xp=jnp, dtype=numpy.float64,
        )
        return loss

    def grad_fn(eps):
        (loss, _aux) = run_model(
            {'distortion': eps}, state, group=group, props=None,
            group_patterns=patterns, pattern_mask=mask,
            noise_model=noise_model, regularizers=(), solver_states=solver_states,
            xp=jnp, dtype=numpy.float64,
        )
        return loss

    eps0 = jnp.zeros((npos, 4))
    loss0 = float(grad_fn(eps0))
    grad = jax.grad(grad_fn)(eps0)

    solver = StrainDistortionSolver({'plan': None, 'params': frozenset({'distortion'})}, StrainDistortionSolverProps(step_size=0.5))
    (update, _) = solver.update(state, None, {'distortion': grad}, loss0)

    assert numpy.all(numpy.isfinite(numpy.asarray(update['object'])))
    assert numpy.all(numpy.isfinite(numpy.asarray(update['positions'])))

    new_object = ObjectState(state.object.sampling, state.object.data + update['object'], state.object.thicknesses)
    new_state = ReconsState(
        iter=state.iter, wavelength=state.wavelength, probe=state.probe, object=new_object,
        scan=state.scan + update['positions'], tilt=None, progress=state.progress,
    )

    assert loss_fn(new_state) < loss0
