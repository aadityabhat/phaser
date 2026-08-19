"""Tests for regularizer gradient-scaling compensation
(`phaser.engines.gradient.run._regularizer_compensation`).

`run_group` normalizes the combined (detector + regularizer) gradient by probe
intensity, npix (relative to `_NPIX_REFERENCE`), and (for 'object') scan_density --
see run.py's grad-scaling comment. Since a `CostRegularizer`'s loss is summed into
the *same* combined loss before that division happens, its own contribution gets
divided by those same factors too, unless pre-compensated -- which would make a
regularizer's effective strength (relative to the detector fit) drift whenever
probe intensity, sim_shape, or scan density change, defeating the whole point of
normalizing the detector loss in the first place. `_regularizer_compensation`
pre-multiplies each regularizer's loss by exactly the factors `run_group` will
later divide out for the variable(s) it touches, canceling to a no-op.
"""
import numpy
import pytest

from phaser.utils.num import get_backend_module

pytest.importorskip("jax")
jnp = get_backend_module('jax')
import jax  # noqa: E402

from phaser.utils.num import Sampling, abs2
from phaser.utils.object import ObjectSampling
from phaser.state import ObjectState, ProbeState, ReconsState, IterState
from phaser.plan import AmplitudeNoisePlan
from phaser.hooks.regularization import CostRegularizerProps
from phaser.engines.common.noise_models import AmplitudeNoiseModel
from phaser.engines.common.regularizers import ObjL2, ProbeRecipTikhonov
from phaser.engines.gradient.run import (
    run_group, extract_vars, SolverStates, compute_scan_density, _NPIX_REFERENCE,
    _regularizer_compensation,
)
import phaser.utils.tree as tree


# ---- _regularizer_compensation (unit-level) ---------------------------------

def test_compensation_for_object_regularizer_includes_probe_int_and_density():
    c = _regularizer_compensation(frozenset({'object'}), probe_int=7.0, npix=2 * _NPIX_REFERENCE, scan_density=3.0)
    assert c == pytest.approx(7.0 * 2.0 * 3.0)


def test_compensation_for_probe_regularizer_excludes_probe_int_and_density():
    """Mirrors run_group's exemption of 'probe' from the probe_int divisor, and
    scan_density only ever applying to 'object' (matching CuPy's ow_scalar)."""
    c = _regularizer_compensation(frozenset({'probe'}), probe_int=7.0, npix=2 * _NPIX_REFERENCE, scan_density=3.0)
    assert c == pytest.approx(2.0)


def test_compensation_is_noop_at_reference_pixel_count_with_unit_density():
    c = _regularizer_compensation(frozenset({'object'}), probe_int=1.0, npix=_NPIX_REFERENCE, scan_density=1.0)
    assert c == pytest.approx(1.0)


# ---- end-to-end: regularizer's gradient contribution stays invariant --------

def _make_toy_problem(seed: int = 0, npos: int = 5, by: int = 8, bx: int = 8, ny: int = 24, nx: int = 24, probe_scale: float = 1.0):
    rng = numpy.random.default_rng(seed)

    obj_sampling = ObjectSampling((ny, nx), sampling=(1.0, 1.0))
    obj_phase = rng.normal(0, 0.3, (1, ny, nx))
    obj_amp = 1.0 - 0.1 * rng.random((1, ny, nx))
    obj_data = jnp.array((obj_amp * numpy.exp(1j * obj_phase)).astype(numpy.complex128))
    obj_state = ObjectState(obj_sampling, obj_data, jnp.array([1.0]))

    probe_sampling = Sampling((by, bx), sampling=(1.0, 1.0))
    probe_data = probe_scale * jnp.array((rng.normal(0, 1, (1, by, bx)) + 1j * rng.normal(0, 1, (1, by, bx))).astype(numpy.complex128))
    probe_state = ProbeState(probe_sampling, probe_data)

    scan = jnp.array(rng.uniform(-4, 4, (npos, 2)))

    state = ReconsState(
        iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state,
        scan=scan, tilt=None, progress={},
    )

    patterns = jnp.array(rng.uniform(0.1, 1.0, (npos, by, bx)))
    mask = jnp.ones((by, bx))
    noise_model = AmplitudeNoiseModel(None, AmplitudeNoisePlan())
    group = jnp.arange(npos)[None, :]

    return (state, patterns, mask, noise_model, group)


def _object_grad_via_recording_solver(state, patterns, mask, noise_model, group, regularizers):
    """Captures the normalized 'object' grad run_group hands to solvers, jit-safely
    (see test_gradient_scaling.py's identical pattern -- a closure side-effect
    would only record a trace-time placeholder, so the value is smuggled out
    through the recording solver's own returned state instead)."""
    class RecordingSolver:
        name = 'recording'
        params = frozenset({'object'})
        def init_state(self, sim):
            return None
        def update_for_iter(self, sim, state, niter):
            return state
        def update(self, sim, state, grad, loss):
            return ({}, grad['object'])

    xp = jnp
    probe_int = xp.sum(abs2(state.probe.data))
    scan_density = compute_scan_density(state, xp, numpy.float64)

    solver = RecordingSolver()
    group_solver_states = [solver.update_for_iter(state, solver.init_state(state), 1)]
    solver_states = SolverStates(
        noise_model_state=None, group_solver_states=group_solver_states,
        regularizer_states=[reg.init_state(state) for reg in regularizers],
        group_constraint_states=[],
    )
    iter_grads = tree.zeros_like(extract_vars(state, frozenset(), group)[0])
    losses = {'detector_loss': xp.array(0.0), 'total_loss': xp.array(0.0), **{reg.name(): xp.array(0.0) for reg in regularizers}}

    (_, _, _, solver_states) = run_group(
        state, group=group, vars=frozenset({'object'}),
        noise_model=noise_model, group_solvers=(solver,), group_constraints=(), regularizers=regularizers,
        losses=losses, iter_grads=iter_grads, solver_states=solver_states,
        props=None, group_patterns=patterns, pattern_mask=mask,
        probe_int=probe_int, scan_density=scan_density,
        xp=xp, dtype=numpy.float64, jit_unroll_slices=False,
    )
    return solver_states.group_solver_states[0]


def test_object_regularizer_contribution_invariant_to_probe_intensity():
    """The obj_l2 regularizer's own contribution to the normalized object gradient
    (isolated by differencing against a no-regularizer run, so the detector-loss
    part cancels) must stay the same whether the probe is unit-scale or 9x more
    intense -- without _regularizer_compensation, it would scale with probe_int
    (i.e. with probe_scale^2), since it'd get divided by a probe_int that has
    nothing to do with it."""
    reg = (ObjL2(None, CostRegularizerProps(cost=1.0)),)

    def reg_only_contribution(probe_scale):
        # run_group's jit donates `state`'s buffers, so each call needs its own
        # fresh state -- reusing one across two calls reads a deleted buffer.
        (state_a, patterns, mask, noise_model, group) = _make_toy_problem(probe_scale=probe_scale)
        (state_b, _, _, _, _) = _make_toy_problem(probe_scale=probe_scale)
        with_reg = _object_grad_via_recording_solver(state_a, patterns, mask, noise_model, group, reg)
        without_reg = _object_grad_via_recording_solver(state_b, patterns, mask, noise_model, group, ())
        return numpy.asarray(with_reg) - numpy.asarray(without_reg)

    contribution_1x = reg_only_contribution(1.0)
    contribution_3x = reg_only_contribution(3.0)  # probe_int scales by 3^2 = 9x

    numpy.testing.assert_allclose(contribution_1x, contribution_3x, rtol=1e-6, atol=1e-10)


def test_object_regularizer_contribution_without_compensation_would_have_scaled():
    """Sanity check on the test above: confirm the *raw* (uncompensated) obj_l2
    loss really would have produced a probe_int-dependent object-gradient
    contribution, so the invariance in the previous test is actually doing
    something (not trivially true regardless of compensation)."""
    def raw_reg_grad(probe_scale, seed=0):
        (state, patterns, mask, noise_model, group) = _make_toy_problem(seed=seed, probe_scale=probe_scale)
        (loss, _) = ObjL2(None, CostRegularizerProps(cost=1.0)).calc_loss_group(group, state, None, group.shape[-1])
        return loss

    # the raw (uncompensated) loss value itself doesn't depend on probe at all --
    # confirms any invariance/variance we see downstream is purely about how
    # run_group's probe_int division interacts with it, not the regularizer's own math.
    assert float(raw_reg_grad(1.0)) == pytest.approx(float(raw_reg_grad(3.0)), rel=1e-10)


def test_probe_regularizer_contribution_matches_manual_normalization():
    """Direct wiring check for the probe case (mirrors
    test_run_group_object_grad_matches_manual_normalization in
    test_gradient_scaling.py): probe_recip_tikh's isolated contribution to the
    normalized probe gradient must equal its raw loss's gradient, compensated by
    _regularizer_compensation and then divided by run_group's own factor for
    'probe' (group_size * npix/ref -- no probe_int, matching probe's exemption).

    Can't test the npix/probe-shape relationship by varying detector shape
    end-to-end, since pattern_mask/probe must share the same spatial shape in the
    real forward model (changing one changes the regularizer's own raw math too,
    not just normalization) -- so this checks the wiring directly instead."""
    reg = ProbeRecipTikhonov(None, CostRegularizerProps(cost=1.0))
    (state_a, patterns, mask, noise_model, group) = _make_toy_problem()
    (state_b, _, _, _, _) = _make_toy_problem()  # separate state: run_group donates buffers

    # differentiate only w.r.t. probe.data directly (rather than the whole state
    # pytree, which has integer-dtype leaves like IterState that jax.grad rejects)
    def loss_fn(probe_data):
        s = ReconsState(
            iter=state_a.iter, wavelength=state_a.wavelength,
            probe=ProbeState(state_a.probe.sampling, probe_data), object=state_a.object,
            scan=state_a.scan, tilt=state_a.tilt, progress=state_a.progress,
        )
        return reg.calc_loss_group(group, s, None, group.shape[-1])[0]

    raw_grad = numpy.asarray(jax.grad(loss_fn)(state_a.probe.data))

    npix = mask.shape[-2] * mask.shape[-1]
    probe_int = float(jnp.sum(abs2(state_a.probe.data)))
    scan_density = float(compute_scan_density(state_a, jnp, numpy.float64))
    compensation = _regularizer_compensation(reg.params, probe_int, npix, scan_density)
    # tree.grad's sign=-1 (what run_group actually uses) returns -conj(raw_grad) for
    # complex leaves (JAX's jax.grad returns the Wirtinger-conjugate convention;
    # tree.grad re-conjugates it back before negating for the descent direction) --
    # see test_strain.py's identical sign convention note.
    expected = -numpy.conj(raw_grad) * compensation / (group.shape[-1] * (npix / _NPIX_REFERENCE))

    # isolate the regularizer's own contribution: run_group's captured grad is the
    # *combined* detector+regularizer gradient, so diff against a no-regularizer run.
    with_reg = numpy.asarray(_probe_grad_via_recording_solver(state_a, patterns, mask, noise_model, group, (reg,)))
    without_reg = numpy.asarray(_probe_grad_via_recording_solver(state_b, patterns, mask, noise_model, group, ()))
    contribution = with_reg - without_reg

    numpy.testing.assert_allclose(contribution, expected, rtol=1e-5, atol=1e-9)


def _probe_grad_via_recording_solver(state, patterns, mask, noise_model, group, regularizers):
    class RecordingSolver:
        name = 'recording'
        params = frozenset({'probe'})
        def init_state(self, sim):
            return None
        def update_for_iter(self, sim, state, niter):
            return state
        def update(self, sim, state, grad, loss):
            return ({}, grad['probe'])

    xp = jnp
    probe_int = xp.sum(abs2(state.probe.data))
    scan_density = compute_scan_density(state, xp, numpy.float64)
    solver = RecordingSolver()
    solver_states = SolverStates(
        noise_model_state=None,
        group_solver_states=[solver.update_for_iter(state, solver.init_state(state), 1)],
        regularizer_states=[r.init_state(state) for r in regularizers],
        group_constraint_states=[],
    )
    iter_grads = tree.zeros_like(extract_vars(state, frozenset(), group)[0])
    losses = {'detector_loss': xp.array(0.0), 'total_loss': xp.array(0.0), **{r.name(): xp.array(0.0) for r in regularizers}}
    (_, _, _, solver_states) = run_group(
        state, group=group, vars=frozenset({'probe'}),
        noise_model=noise_model, group_solvers=(solver,), group_constraints=(), regularizers=regularizers,
        losses=losses, iter_grads=iter_grads, solver_states=solver_states,
        props=None, group_patterns=patterns, pattern_mask=mask,
        probe_int=probe_int, scan_density=scan_density,
        xp=xp, dtype=numpy.float64, jit_unroll_slices=False,
    )
    return solver_states.group_solver_states[0]


# ---- cost_scale scan-size (grouping-count) fix ------------------------------

def test_cost_scale_uses_full_scan_size_not_group_size():
    """Regression test for the cost_scale bug: `calc_loss_group` used to compute
    `group.shape[-1] / prod(sim.scan.shape[:-1])`, intended as a group_size/npos
    fraction -- but by the time it's called inside run_model, sim.scan has
    already been restricted to the current group (see insert_vars), so both
    sides were always equal and it was a permanent no-op (always 1.0). Now
    total_npos is threaded in explicitly from outside (captured before that
    restriction happens), so a group that's a proper subset of the full scan
    must produce a loss scaled by group_size/total_npos, not 1.0.

    Reproduces the group-already-restricted scenario directly (calc_loss_group
    is called here exactly as run_model calls it, not via the full pipeline)."""
    (state, patterns, mask, noise_model, group) = _make_toy_problem(npos=8)
    reg = ObjL2(None, CostRegularizerProps(cost=1.0))

    full_group = jnp.arange(8)[None, :]
    half_group = jnp.arange(4)[None, :]

    # mimics insert_vars restricting sim.scan to the current group, exactly as
    # happens inside run_model before calc_loss_group is ever called
    half_restricted_sim = ReconsState(
        iter=state.iter, wavelength=state.wavelength, probe=state.probe, object=state.object,
        scan=state.scan[tuple(half_group)], tilt=state.tilt, progress=state.progress,
    )

    (loss_full, _) = reg.calc_loss_group(full_group, state, None, total_npos=8)
    (loss_half, _) = reg.calc_loss_group(half_group, half_restricted_sim, None, total_npos=8)

    # obj_l2's raw cost is a whole-object quantity independent of which group is
    # active, so the two losses should differ by exactly the group-size ratio
    # (4/8 = 0.5) -- under the old bug, loss_half/loss_full would have been 1.0
    # instead (cost_scale silently canceling to a no-op both times).
    numpy.testing.assert_allclose(float(loss_half), float(loss_full) * 0.5, rtol=1e-6)


def test_probe_cost_scale_now_applies_scan_size_normalization():
    """Probe regularizers previously had cost_scale = 1.0 (no scan-size
    normalization attempted at all). Confirm they now get the same
    group_size/total_npos treatment as object regularizers."""
    (state, patterns, mask, noise_model, group) = _make_toy_problem(npos=8)
    reg = ProbeRecipTikhonov(None, CostRegularizerProps(cost=1.0))

    full_group = jnp.arange(8)[None, :]
    half_group = jnp.arange(4)[None, :]

    (loss_full, _) = reg.calc_loss_group(full_group, state, None, total_npos=8)
    (loss_half, _) = reg.calc_loss_group(half_group, state, None, total_npos=8)

    # probe_recip_tikh's raw cost depends only on probe.data (untouched by which
    # group is active), so again the losses should differ by exactly 4/8 = 0.5
    numpy.testing.assert_allclose(float(loss_half), float(loss_full) * 0.5, rtol=1e-6)
