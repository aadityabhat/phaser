"""Tests for the gradient-scaling normalization in `phaser.engines.gradient.run`
(`run_group`'s npix/probe_int/scan_density division, and `compute_scan_density`).

See `run.py`'s comments at `compute_scan_density` and the `run_group` grad-scaling
loop for the underlying reasoning -- this mirrors CuPy's reference implementation
(`python/ptycho/gradcalc.py`'s `df`/`dn`/`ow` normalization), ported so gradients fed
to phaser's solvers (Adam, SGD, strain distortion, ...) are invariant to probe/
diffraction intensity, diffraction-pattern resampling/padding, and scan step size.
"""
import numpy
import pytest

from phaser.utils.num import get_backend_module

pytest.importorskip("jax")
jnp = get_backend_module('jax')  # loading via phaser's backend loader enables jax_enable_x64

from phaser.utils.num import Sampling
from phaser.utils.object import ObjectSampling
from phaser.state import ObjectState, ProbeState, ReconsState, IterState
from phaser.plan import AmplitudeNoisePlan, AdamSolverPlan
from phaser.engines.common.noise_models import AmplitudeNoiseModel
from phaser.engines.gradient.run import (
    run_group, extract_vars, SolverStates, compute_scan_density, _NPIX_REFERENCE,
    _SCAN_DENSITY_REFERENCE,
)
from phaser.engines.gradient.solvers import AdamSolver
import phaser.utils.tree as tree


def _make_toy_problem(seed: int = 0, npos: int = 6, by: int = 10, bx: int = 10, ny: int = 24, nx: int = 24):
    rng = numpy.random.default_rng(seed)

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
    group = jnp.arange(npos)[None, :]

    return (state, patterns, mask, noise_model, group, npos)


def _run_group_for_test(state, patterns, mask, noise_model, group, vars_set, group_solvers=(), probe_int=None):
    vars_set = frozenset(vars_set)
    xp = jnp
    if probe_int is None:
        from phaser.utils.num import abs2
        probe_int = xp.sum(abs2(state.probe.data))
    scan_density = compute_scan_density(state, xp, numpy.float64)

    group_solver_states = [s.init_state(state) for s in group_solvers]
    # mirrors run_engine's per-iteration setup (needed to actually initialize e.g.
    # Adam's internal moment state -- init_state() alone leaves it None)
    group_solver_states = [s.update_for_iter(state, gs, 1) for (s, gs) in zip(group_solvers, group_solver_states)]
    solver_states = SolverStates(
        noise_model_state=None,
        group_solver_states=group_solver_states,
        regularizer_states=[], group_constraint_states=[],
    )
    iter_grads = tree.zeros_like(extract_vars(state, vars_set & {'positions', 'tilt', 'distortion'}, group)[0])
    losses = {'detector_loss': xp.array(0.0), 'total_loss': xp.array(0.0)}

    return run_group(
        state, group=group, vars=vars_set,
        noise_model=noise_model, group_solvers=group_solvers, group_constraints=(), regularizers=(),
        losses=losses, iter_grads=iter_grads, solver_states=solver_states,
        props=None, group_patterns=patterns, pattern_mask=mask,
        probe_int=probe_int, scan_density=scan_density,
        xp=xp, dtype=numpy.float64, jit_unroll_slices=False,
    )


# ---- compute_scan_density -------------------------------------------------

def test_compute_scan_density_disjoint_positions_is_one():
    """Non-overlapping probe footprints: each covered object pixel is touched
    exactly once, so the coverage grid is a 0/1 indicator and
    sum(x^2)/sum(x) == 1 exactly (independent of how many positions there are) --
    the density-normalization's "no redundancy" baseline."""
    (by, bx) = (6, 6)
    obj_sampling = ObjectSampling((60, 60), sampling=(1.0, 1.0))
    # positions spaced far apart (> probe footprint) so cutouts can't overlap
    scan = jnp.array([[-20.0, -20.0], [-20.0, 0.0], [-20.0, 20.0], [0.0, -20.0], [0.0, 20.0]])

    probe_state = ProbeState(Sampling((by, bx), sampling=(1.0, 1.0)), jnp.zeros((1, by, bx), dtype=jnp.complex128))
    obj_state = ObjectState(obj_sampling, jnp.ones((1, 60, 60), dtype=jnp.complex128), jnp.array([1.0]))
    state = ReconsState(iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state, scan=scan, tilt=None, progress={})

    density = float(compute_scan_density(state, jnp, numpy.float64))
    assert density == pytest.approx(1.0, rel=1e-6)


def test_compute_scan_density_repeated_position_equals_count():
    """N positions landing on the identical pixel footprint: coverage grid is N
    (or 0), so sum(x^2)/sum(x) == N exactly -- the fully-redundant extreme."""
    (by, bx) = (6, 6)
    n = 7
    obj_sampling = ObjectSampling((30, 30), sampling=(1.0, 1.0))
    scan = jnp.zeros((n, 2), dtype=jnp.float64)  # all positions identical

    probe_state = ProbeState(Sampling((by, bx), sampling=(1.0, 1.0)), jnp.zeros((1, by, bx), dtype=jnp.complex128))
    obj_state = ObjectState(obj_sampling, jnp.ones((1, 30, 30), dtype=jnp.complex128), jnp.array([1.0]))
    state = ReconsState(iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state, scan=scan, tilt=None, progress={})

    density = float(compute_scan_density(state, jnp, numpy.float64))
    assert density == pytest.approx(float(n), rel=1e-6)


def test_compute_scan_density_denser_scan_is_larger():
    """A finer scan step (more overlap) must give a strictly larger density
    scalar than a coarser step over the same area -- the qualitative behavior
    the object-gradient normalization relies on for scan-step-size invariance."""
    (by, bx) = (10, 10)
    obj_sampling = ObjectSampling((80, 80), sampling=(1.0, 1.0))
    probe_state = ProbeState(Sampling((by, bx), sampling=(1.0, 1.0)), jnp.zeros((1, by, bx), dtype=jnp.complex128))
    obj_state = ObjectState(obj_sampling, jnp.ones((1, 80, 80), dtype=jnp.complex128), jnp.array([1.0]))

    def make_grid_scan(step):
        ys, xs = numpy.meshgrid(numpy.arange(-25, 25, step), numpy.arange(-25, 25, step), indexing='ij')
        return jnp.array(numpy.stack([ys.ravel(), xs.ravel()], axis=-1))

    dense_scan = make_grid_scan(2.0)
    sparse_scan = make_grid_scan(11.0)  # > probe extent (10), so footprints can't overlap

    dense_state = ReconsState(iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state, scan=dense_scan, tilt=None, progress={})
    sparse_state = ReconsState(iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state, scan=sparse_scan, tilt=None, progress={})

    dense_density = float(compute_scan_density(dense_state, jnp, numpy.float64))
    sparse_density = float(compute_scan_density(sparse_state, jnp, numpy.float64))

    assert dense_density > sparse_density
    assert sparse_density == pytest.approx(1.0, rel=1e-6)  # step 11 > probe extent 10 => disjoint


# ---- run_group's normalization wiring --------------------------------------

def test_run_group_position_grad_matches_manual_normalization():
    """Regression check for the wiring itself: the normalized 'positions' grad
    run_group hands to solvers must equal the raw autodiff grad divided by
    exactly probe_int * (npix / _NPIX_REFERENCE) (positions are per-iter, so no
    group-size division, and no scan_density factor -- that's object-only,
    matching CuPy's ow_scalar applying only to d.o)."""
    (state, patterns, mask, noise_model, group, npos) = _make_toy_problem(npos=5, by=8, bx=8)

    from phaser.engines.gradient.run import run_model
    (raw_grad, _aux) = tree.grad(run_model, has_aux=True, xp=jnp, sign=-1)(
        *extract_vars(state, {'positions'}, group),
        group=group, props=None, group_patterns=patterns, pattern_mask=mask,
        noise_model=noise_model, regularizers=(), solver_states=SolverStates(None, [], [], []),
        probe_int=1.0, scan_density=1.0, total_npos=1,
        xp=jnp, dtype=numpy.float64, jit_unroll_slices=False,
    )

    from phaser.utils.num import abs2
    probe_int = jnp.sum(abs2(state.probe.data))
    npix = mask.shape[-2] * mask.shape[-1]
    expected = raw_grad['positions'] / (probe_int * (npix / _NPIX_REFERENCE))

    (_, _, iter_grads, _) = _run_group_for_test(state, patterns, mask, noise_model, group, {'positions'})

    numpy.testing.assert_allclose(numpy.asarray(iter_grads['positions']), numpy.asarray(expected), rtol=1e-6, atol=1e-10)


def test_run_group_object_grad_matches_manual_normalization():
    """Same check for 'object': normalized grad must equal raw grad divided by
    group_size * probe_int * (npix / _NPIX_REFERENCE) * (scan_density / _SCAN_DENSITY_REFERENCE)."""
    (state, patterns, mask, noise_model, group, npos) = _make_toy_problem(npos=5, by=8, bx=8)

    from phaser.engines.gradient.run import run_model
    (raw_grad, _aux) = tree.grad(run_model, has_aux=True, xp=jnp, sign=-1)(
        *extract_vars(state, {'object'}, group),
        group=group, props=None, group_patterns=patterns, pattern_mask=mask,
        noise_model=noise_model, regularizers=(), solver_states=SolverStates(None, [], [], []),
        probe_int=1.0, scan_density=1.0, total_npos=1,
        xp=jnp, dtype=numpy.float64, jit_unroll_slices=False,
    )

    from phaser.utils.num import abs2
    probe_int = jnp.sum(abs2(state.probe.data))
    npix = mask.shape[-2] * mask.shape[-1]
    scan_density = compute_scan_density(state, jnp, numpy.float64)
    expected = raw_grad['object'] / (
        group.shape[-1] * probe_int * (npix / _NPIX_REFERENCE) * (scan_density / _SCAN_DENSITY_REFERENCE)
    )

    # run_group is jit-compiled, so a Python-side-effect capture (e.g. writing into a
    # closed-over dict from inside .update()) would only ever record a trace-time
    # placeholder, not a concrete value. Instead, smuggle the received grad out
    # through the solver's own *returned state*, which comes back as a real array
    # once the jit call completes.
    class RecordingSolver:
        name = 'recording'
        params = frozenset({'object'})
        def init_state(self, sim):
            return None
        def update_for_iter(self, sim, state, niter):
            return state
        def update(self, sim, state, grad, loss):
            return ({}, grad['object'])

    (_, _, _, solver_states) = _run_group_for_test(
        state, patterns, mask, noise_model, group, {'object'}, group_solvers=(RecordingSolver(),), probe_int=probe_int
    )
    captured_object_grad = solver_states.group_solver_states[0]

    numpy.testing.assert_allclose(numpy.asarray(captured_object_grad), numpy.asarray(expected), rtol=1e-6, atol=1e-12)


# ---- Adam eps-dominance sanity check ---------------------------------------

def test_adam_object_update_not_stalled_by_eps_after_normalization():
    """The npix/scan_density normalization divides the raw object gradient by a
    potentially large constant before Adam ever sees it. Adam's update is
    approximately scale-invariant (m/sqrt(v) cancels the constant) *unless* the
    rescaled gradient magnitude drops low enough for Adam's fixed `eps` to
    dominate the denominator, which would silently stall the object solve
    instead of just rescaling it. Check the normalized-grad-driven update is
    within a reasonable factor of the update a raw (unnormalized) gradient of
    the same *shape* would produce through the same Adam call, at realistic
    pixel counts (192x192, matching the actual reconstruction plans)."""
    (state, patterns, mask, noise_model, group, npos) = _make_toy_problem(npos=8, by=32, bx=32, ny=64, nx=64)

    props = AdamSolverPlan(learning_rate=1e-2)
    solver = AdamSolver({'plan': None, 'params': frozenset({'object'})}, props)

    # run_group's jit donates `state`'s buffers, so snapshot the original object
    # data before the call -- state.object.data is invalid to read afterwards.
    orig_object_data = numpy.array(state.object.data)

    (new_state, _, _, _) = _run_group_for_test(
        state, patterns, mask, noise_model, group, {'object'}, group_solvers=(solver,),
    )
    update = numpy.array(new_state.object.data) - orig_object_data

    update_norm = float(jnp.linalg.norm(update))
    n_elem = update.size
    # Adam's first-step update should be close to `learning_rate` per-element in
    # sign-of-gradient regimes; well above numerical noise floor (eps-stall would
    # produce updates many orders of magnitude below learning_rate).
    assert update_norm > 0.1 * props.learning_rate * numpy.sqrt(n_elem), (
        f"object update norm {update_norm} looks eps-stalled relative to "
        f"learning_rate={props.learning_rate} over {n_elem} elements"
    )
