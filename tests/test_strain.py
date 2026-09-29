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
from phaser.engines.gradient.run import run_model, SolverStates, apply_update
from phaser.engines.common.strain import (
    StrainDistortionSolver, StrainDistortionSolverProps, strain_perturbation, _scattered_kernel_regression,
    solve_distortion,
)
from phaser.engines.common.simulation import make_propagators
from phaser.utils.num import invavg
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
        probe_int=1.0, scan_density=1.0, total_npos=1,
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
    # total_iter must be > 1 for the solver to apply a nonzero step at all -- see
    # test_strain_step_schedule_skips_first_iteration -- and <= 250 (the default
    # warmup_freeze_gate_iter) to exercise the plain decayed/normalized step, not the
    # running-average-then-freeze path (covered separately).
    state.iter.total_iter = 5

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    eps0 = jnp.zeros((npos, 4))
    loss0 = float(loss_fn(eps0))
    descent_grad = tree.grad(loss_fn, xp=jnp, sign=-1)(eps0)

    solver = StrainDistortionSolver({'plan': None, 'params': frozenset({'distortion'})}, StrainDistortionSolverProps(step_size=0.5))
    (update, _) = solver.update(state, solver.init_state(state), {'distortion': descent_grad}, loss0)

    assert numpy.all(numpy.isfinite(numpy.asarray(update['object'])))
    assert numpy.all(numpy.isfinite(numpy.asarray(update['positions'])))

    new_object = ObjectState(state.object.sampling, state.object.data + update['object'], state.object.thicknesses)
    new_state = ReconsState(
        iter=state.iter, wavelength=state.wavelength, probe=state.probe, object=new_object,
        scan=state.scan + update['positions'], tilt=None, progress=state.progress,
    )

    loss_after = float(_loss(new_state, patterns, mask, noise_model, solver_states, group, {}))
    assert loss_after < loss0


def test_apply_update_subtracts_mean_for_standalone_positions():
    """`apply_update`'s translation gauge-fixing (subtracting the mean position update) is
    meant for a plain per-position gradient solver (e.g. Adam/SGD on 'positions' alone),
    which has no other way to pin down the reconstruction's otherwise-unconstrained global
    translation. A standalone {'positions': ...} update (no paired 'object' delta) should
    have this applied."""
    (state, _patterns, _mask, _noise_model, _solver_states, _group, npos) = _make_toy_problem()
    delta = jnp.array(numpy.random.default_rng(0).normal(size=(npos, 2)))
    scan_before = numpy.array(state.scan)

    new_state = apply_update(state, {'positions': delta})
    applied = numpy.asarray(new_state.scan) - scan_before

    numpy.testing.assert_allclose(numpy.mean(applied, axis=0), 0.0, atol=1e-10)
    assert not numpy.allclose(applied, numpy.asarray(delta))


def test_apply_update_preserves_strain_distortion_positions_mean():
    """Regression test: `apply_update` used to unconditionally mean-subtract every
    {'positions': ...} update, including `StrainDistortionSolver`'s -- but that update is
    produced together with a correlated 'object' delta from the same Poisson-solved
    displacement field (see `StrainDistortionSolver.update`/`solve_distortion`), which
    already fixes its own gauge freedom (`solve_distortion`'s `zero_mask` zeros the field's
    DC component). Re-subtracting the mean here decoupled the position shift from the object
    warp it's paired with -- the CuPy reference implementation's `distort()` applies no such
    post-hoc correction to `sd`. A coupled {'object', 'positions'} update should pass the
    positions delta through unchanged."""
    (state, _patterns, _mask, _noise_model, _solver_states, _group, npos) = _make_toy_problem()
    delta_pos = jnp.array(numpy.random.default_rng(1).normal(size=(npos, 2)))
    delta_obj = jnp.zeros_like(state.object.data)
    scan_before = numpy.array(state.scan)

    new_state = apply_update(state, {'object': delta_obj, 'positions': delta_pos})
    applied = numpy.asarray(new_state.scan) - scan_before

    numpy.testing.assert_allclose(applied, numpy.asarray(delta_pos), atol=1e-10)


def _unit_positions_delta(state, raw_grad, min_neighbors=6):
    """`solve_distortion`'s position update at step_scale=1 -- since the kernel regression,
    Poisson solve, and position sampling are all linear in `delta_eps` (only the object's
    pull-warp is not), `StrainDistortionSolver`'s actual positions_delta at any step_scale
    should equal `step_scale * this`."""
    corner = state.object.sampling.corner
    sampling = state.object.sampling.sampling
    scan_idx = (state.scan - corner) / sampling
    (_, new_scan_idx) = solve_distortion(scan_idx, state.object.data, raw_grad, min_neighbors=min_neighbors)
    new_scan = new_scan_idx * sampling + corner
    return numpy.asarray(new_scan - state.scan)


def test_strain_step_schedule_skips_first_iteration():
    """Ported from CuPy's `always_apply_dsdf or i > 1` gate (`python/ptycho/optimize.py`): a
    fresh reconstruction's very first iteration has no prior gradient history, so the solver
    should apply no update at all, rather than normalize off a single, possibly degenerate
    sample."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()
    state.iter.total_iter = 1

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    descent_grad = tree.grad(loss_fn, xp=jnp, sign=-1)(jnp.zeros((npos, 4)))
    solver = StrainDistortionSolver({'plan': None, 'params': frozenset({'distortion'})}, StrainDistortionSolverProps(step_size=0.5))
    (update, _) = solver.update(state, solver.init_state(state), {'distortion': descent_grad}, 0.0)

    numpy.testing.assert_allclose(numpy.asarray(update['object']), 0.0, atol=1e-12)
    numpy.testing.assert_allclose(numpy.asarray(update['positions']), 0.0, atol=1e-9)


def test_strain_step_schedule_decays_with_iteration():
    """Ported from CuPy's `0.5 ** (i / 1000)` decay on `df_raw` (`python/ptycho/optimize.py`):
    the step size at total_iter=1002 should be half of total_iter=2's (both well clear of the
    default warmup_freeze_gate_iter=250, so disable freezing here to isolate the decay)."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    descent_grad = tree.grad(loss_fn, xp=jnp, sign=-1)(jnp.zeros((npos, 4)))
    props = StrainDistortionSolverProps(step_size=0.5, warmup_freeze_gate_iter=None, decay_half_life=1000.0)
    solver = StrainDistortionSolver({'plan': None, 'params': frozenset({'distortion'})}, props)

    state.iter.total_iter = 2
    (update_early, _) = solver.update(state, solver.init_state(state), {'distortion': descent_grad}, 0.0)
    state.iter.total_iter = 1002
    (update_late, _) = solver.update(state, solver.init_state(state), {'distortion': descent_grad}, 0.0)

    ratio = numpy.linalg.norm(numpy.asarray(update_late['positions'])) / numpy.linalg.norm(numpy.asarray(update_early['positions']))
    numpy.testing.assert_allclose(ratio, 0.5, rtol=0.02)


def test_strain_step_schedule_warmup_then_freezes():
    """Ported from CuPy's SCAN_STEP_WARMUP_ITERS/`i > 250` gate (`python/ptycho/optimize.py`):
    past `warmup_freeze_gate_iter`, the step size is the running average of its own raw
    (decayed, invavg-normalized) value over the next `warmup_freeze_iters` iterations, then
    frozen at that average -- plain fixed-step descent from there on, even as the (disabled-
    after-freeze) decay would otherwise keep shrinking it."""
    (state, patterns, mask, noise_model, solver_states, group, npos) = _make_toy_problem()

    def loss_fn(eps):
        return _loss(state, patterns, mask, noise_model, solver_states, group, {'distortion': eps})

    descent_grad = tree.grad(loss_fn, xp=jnp, sign=-1)(jnp.zeros((npos, 4)))
    props = StrainDistortionSolverProps(
        step_size=0.5, decay_half_life=10.0, warmup_freeze_gate_iter=3, warmup_freeze_iters=3,
    )
    solver = StrainDistortionSolver({'plan': None, 'params': frozenset({'distortion'})}, props)
    solver_state = solver.init_state(state)
    unit_delta = _unit_positions_delta(state, descent_grad)
    denom = float(invavg(numpy.asarray(descent_grad)))

    expected_running_avg = 0.0
    frozen_expected = None
    for (k, total_iter) in enumerate((4, 5, 6, 7, 8), start=1):
        state.iter.total_iter = total_iter
        (update, solver_state) = solver.update(state, solver_state, {'distortion': descent_grad}, 0.0)

        if frozen_expected is None:
            raw = props.step_size * 0.5 ** (total_iter / props.decay_half_life) / denom
            expected_running_avg = (expected_running_avg * (k - 1) + raw) / k
            expected_step_scale = expected_running_avg
            if k >= props.warmup_freeze_iters:
                frozen_expected = expected_step_scale
        else:
            expected_step_scale = frozen_expected

        numpy.testing.assert_allclose(
            numpy.asarray(update['positions']), expected_step_scale * unit_delta, atol=1e-10, rtol=1e-5,
        )

    assert solver_state.frozen == pytest.approx(frozen_expected)


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
        probe_int=1.0, scan_density=1.0, total_npos=1,
        xp=jnp, dtype=numpy.float32, jit_unroll_slices=False,
    )
    assert numpy.isfinite(float(loss))


# ---- _scattered_kernel_regression (2026-09-21 rewrite: Nadaraya-Watson kernel      ----
# ---- regression + backward pull-warp, replacing linear-inside-hull/forward-warp)   ----

def test_kernel_regression_symmetric_input_gives_symmetric_output():
    """Regression test for anchor/off-by-one correctness (see strain.py's module
    docstring -- CuPy's own rewrite had a real 1px anchoring bug in the equivalent
    step). A D4-symmetric input (4 points of equal value, placed symmetrically about
    the grid center) must produce a D4-symmetric output grid on any correctly-indexed
    splat -- a directional indexing bug (e.g. floor vs. floor+1 applied asymmetrically)
    would break this even though it wouldn't be visible in an all-zero/no-op case."""
    grid = (9, 9)
    points = numpy.array([[2.0, 2.0], [2.0, 6.0], [6.0, 2.0], [6.0, 6.0]])
    values = numpy.array([1.0, 1.0, 1.0, 1.0])

    result = numpy.asarray(_scattered_kernel_regression(points, values, grid, min_neighbors=2))

    numpy.testing.assert_allclose(result, result[::-1, :], atol=1e-10)
    numpy.testing.assert_allclose(result, result[:, ::-1], atol=1e-10)
    numpy.testing.assert_allclose(result, result.T, atol=1e-10)
    # and it should actually be spatially structured, not degenerately constant --
    # points near the grid edge (far from all 4 sources) see much less density.
    assert result[0, 0] < result[2, 2]


def test_kernel_regression_numpy_and_jax_agree():
    rng = numpy.random.default_rng(3)
    grid = (10, 10)
    points = rng.uniform(1, 8, (12, 2))
    values = rng.normal(size=(12, 3))

    result_numpy = numpy.asarray(_scattered_kernel_regression(points, values, grid, min_neighbors=3))
    result_jax = numpy.asarray(
        _scattered_kernel_regression(jnp.array(points), jnp.array(values), grid, min_neighbors=3)
    )

    numpy.testing.assert_allclose(result_numpy, result_jax, atol=1e-6, rtol=1e-5)
