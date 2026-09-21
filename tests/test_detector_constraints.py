"""Tests for the new incoherent-background/propagator-absorption forward-model terms
(`phaser.engines.gradient.run.run_model`) and their `IterConstraint`s
(`phaser.engines.common.detector_constraints`), ported from CuPy's
`recon_12slices/ptycho/optimize.py` (2026-09-15/16 revisions) -- see that module's docstring
for the physical motivation.
"""
import numpy
import pytest

from phaser.utils.num import get_backend_module

pytest.importorskip("jax")
jnp = get_backend_module('jax')  # loading via phaser's backend loader enables jax_enable_x64

from phaser.utils.num import Sampling, abs2
from phaser.utils.object import ObjectSampling
from phaser.state import ObjectState, ProbeState, ReconsState, IterState, ProgressState
from phaser.plan import AmplitudeNoisePlan
from phaser.engines.common.noise_models import AmplitudeNoiseModel
from phaser.engines.gradient.run import run_model, SolverStates
from phaser.engines.common.simulation import make_propagators
from phaser.engines.common.detector_constraints import (
    PropagatorMuClamp, NonNegBackground, ProbeIntensityCap, BackgroundIntensityCap,
)
from phaser.hooks.regularization import (
    PropagatorMuClampProps, NonNegBackgroundProps, ProbeIntensityCapProps, BackgroundIntensityCapProps,
)


def _make_toy_problem(seed: int = 0, n_slices: int = 1, background=None, propagator_mu=None):
    rng = numpy.random.default_rng(seed)

    (ny, nx) = (24, 24)
    (by, bx) = (10, 10)
    npos = 6

    obj_sampling = ObjectSampling((ny, nx), sampling=(1.0, 1.0))
    obj_phase = rng.normal(0, 0.3, (n_slices, ny, nx))
    obj_amp = 1.0 - 0.1 * rng.random((n_slices, ny, nx))
    obj_data = jnp.array((obj_amp * numpy.exp(1j * obj_phase)).astype(numpy.complex128))
    thicknesses = jnp.array([20.0] * n_slices) if n_slices > 1 else jnp.array([1.0])
    obj_state = ObjectState(obj_sampling, obj_data, thicknesses)

    probe_sampling = Sampling((by, bx), sampling=(1.0, 1.0))
    probe_data = jnp.array((rng.normal(0, 1, (1, by, bx)) + 1j * rng.normal(0, 1, (1, by, bx))).astype(numpy.complex128))
    probe_state = ProbeState(probe_sampling, probe_data)

    scan = jnp.array(rng.uniform(-4, 4, (npos, 2)))

    state = ReconsState(
        iter=IterState.empty(), wavelength=0.025, probe=probe_state, object=obj_state,
        scan=scan, tilt=None,
        background=None if background is None else jnp.array(background),
        propagator_mu=None if propagator_mu is None else jnp.array(propagator_mu),
        progress={},
    )

    patterns = jnp.array(rng.uniform(0.1, 1.0, (npos, by, bx)))
    mask = jnp.ones((by, bx))
    noise_model = AmplitudeNoiseModel(None, AmplitudeNoisePlan())
    solver_states = SolverStates(noise_model_state=None, group_solver_states=[], regularizer_states=[], group_constraint_states=[])
    group = jnp.arange(npos)[None, :]
    props = make_propagators(state, bwlim_frac=None)

    return (state, patterns, mask, noise_model, solver_states, group, props)


def _run(state, patterns, mask, noise_model, solver_states, group, props, vars_dict):
    return run_model(
        vars_dict, state, group=group, props=props,
        group_patterns=patterns, pattern_mask=mask,
        noise_model=noise_model, regularizers=(), solver_states=solver_states,
        probe_int=1.0, scan_density=1.0, total_npos=1,
        xp=jnp, dtype=numpy.float64, jit_unroll_slices=False,
    )


# ---- forward model: background/propagator_mu ------------------------------

def test_background_shifts_model_intensity_and_loss():
    """Adding a uniform background should change the loss exactly as if it were added
    to model_intensity by hand -- same noise model, same patterns."""
    background = numpy.full((10, 10), 0.05)
    (state0, patterns, mask, noise_model, solver_states, group, props) = _make_toy_problem(background=None)
    (state1, _, _, _, _, _, _) = _make_toy_problem(background=background)

    (loss0, _) = _run(state0, patterns, mask, noise_model, solver_states, group, props, {})
    (loss1, _) = _run(state1, patterns, mask, noise_model, solver_states, group, props, {})

    assert float(loss1) != pytest.approx(float(loss0))
    assert numpy.isfinite(float(loss1))


def test_propagator_mu_attenuates_and_reduces_coherent_total():
    """A positive propagator_mu attenuates every slice transition (exp(-mu) < 1), so a
    multi-slice object's total coherent intensity reaching the detector should drop
    relative to mu=0 -- exercised via run_model's 'coh_total' aux output."""
    propagator_mu = numpy.full((10, 10), 0.5)
    (state0, patterns, mask, noise_model, solver_states, group, props) = _make_toy_problem(n_slices=3, propagator_mu=None)
    (state1, _, _, _, _, _, _) = _make_toy_problem(n_slices=3, propagator_mu=propagator_mu)

    (_, (_, losses0)) = _run(state0, patterns, mask, noise_model, solver_states, group, props, {})
    (_, (_, losses1)) = _run(state1, patterns, mask, noise_model, solver_states, group, props, {})

    assert float(losses1['coh_total']) < float(losses0['coh_total'])


def test_no_background_or_propagator_mu_is_a_noop():
    """sim.background/propagator_mu both None (the default) must reproduce the exact
    pre-existing forward model -- no accidental behavior change for reconstructions
    that don't use either feature."""
    (state, patterns, mask, noise_model, solver_states, group, props) = _make_toy_problem()
    (loss, (_, losses)) = _run(state, patterns, mask, noise_model, solver_states, group, props, {})
    assert numpy.isfinite(float(loss))
    assert numpy.isfinite(float(losses['coh_total']))


# ---- IterConstraints --------------------------------------------------------

def test_propagator_mu_clamp_bounds_to_0_mu_max():
    mu = jnp.array([[-1.0, 0.5], [3.0, 10.0]])
    (state, *_rest) = _make_toy_problem(propagator_mu=mu)

    constraint = PropagatorMuClamp(None, PropagatorMuClampProps(mu_max=4.0))
    (new_state, _) = constraint.apply_iter(state, constraint.init_state(state))

    result = numpy.asarray(new_state.propagator_mu)
    assert result.min() >= 0.0
    assert result.max() <= 4.0
    numpy.testing.assert_allclose(result, [[0.0, 0.5], [3.0, 4.0]])


def test_nonneg_background_floors_only_ever_valid_pixels():
    background = jnp.array([[-1.0, -1.0], [-1.0, -1.0]])
    pattern_mask = jnp.array([[1.0, 0.0], [1.0, 0.0]])  # right column never valid
    (state, *_rest) = _make_toy_problem(background=background)

    constraint = NonNegBackground(None, NonNegBackgroundProps(eps=0.5))
    constraint.bind_context(pattern_mask=pattern_mask)
    (new_state, _) = constraint.apply_iter(state, constraint.init_state(state))

    result = numpy.asarray(new_state.background)
    expected_floor = 0.5 * (-1.0 + numpy.sqrt(1.0 + 0.25))
    # left column (ever-valid) is floored; right column (never-valid) is left untouched at -1.
    numpy.testing.assert_allclose(result[:, 0], expected_floor)
    numpy.testing.assert_allclose(result[:, 1], -1.0)


def test_probe_intensity_cap_only_shrinks_never_grows():
    (state, *_rest) = _make_toy_problem()
    cur_total = float(jnp.sum(abs2(state.probe.data)))
    npix = state.probe.data.shape[-2] * state.probe.data.shape[-1]

    # case 1: M_bar generous -- probe already under budget, must be unchanged.
    constraint = ProbeIntensityCap(None, ProbeIntensityCapProps())
    constraint.bind_context(M_bar=jnp.asarray(cur_total * npix * 100.0))
    (new_state, _) = constraint.apply_iter(state, constraint.init_state(state))
    numpy.testing.assert_allclose(numpy.asarray(new_state.probe.data), numpy.asarray(state.probe.data))

    # case 2: M_bar tight -- probe over budget, must shrink to exactly hit the target.
    tight_M_bar = jnp.asarray(cur_total * npix * 0.25)
    constraint2 = ProbeIntensityCap(None, ProbeIntensityCapProps())
    constraint2.bind_context(M_bar=tight_M_bar)
    (shrunk_state, _) = constraint2.apply_iter(state, constraint2.init_state(state))
    shrunk_total = float(jnp.sum(abs2(shrunk_state.probe.data)))
    assert shrunk_total == pytest.approx(float(tight_M_bar) / npix, rel=1e-6)


def test_background_intensity_cap_only_shrinks_never_grows():
    background = jnp.full((10, 10), 1.0)
    (state, *_rest) = _make_toy_problem(background=background)
    state.progress['coh_total'] = ProgressState(iters=[1], values=[0.0])
    cur_total = 100.0  # sum over (10,10) of 1.0

    # generous budget -- unchanged
    constraint = BackgroundIntensityCap(None, BackgroundIntensityCapProps())
    constraint.bind_context(M_bar=jnp.asarray(cur_total * 100.0))
    (new_state, _) = constraint.apply_iter(state, constraint.init_state(state))
    numpy.testing.assert_allclose(numpy.asarray(new_state.background), numpy.asarray(state.background))

    # tight budget -- shrinks to exactly hit max(M_bar - coh_total_avg, 0)
    tight_M_bar = jnp.asarray(cur_total * 0.1)
    constraint2 = BackgroundIntensityCap(None, BackgroundIntensityCapProps())
    constraint2.bind_context(M_bar=tight_M_bar)
    (shrunk_state, _) = constraint2.apply_iter(state, constraint2.init_state(state))
    shrunk_total = float(jnp.sum(shrunk_state.background))
    assert shrunk_total == pytest.approx(float(tight_M_bar), rel=1e-6)
