"""Tests for `phaser.utils.num.brake` and the `max_step_size` soft-clip wiring on
`SGDSolver` (`phaser.engines.gradient.solvers.scale_by_brake`).

`brake` ports CuPy's `brake()` (`python/ptycho/common.py`) -- a smooth arctan
rolloff that soft-clips a vector's magnitude while preserving direction, used in
CuPy to cap `d.s` (position updates) and now here for both `StrainDistortionSolver`
(replacing its old hard min-clip) and the plain positions SGD solver.
"""
import numpy
import pytest

from phaser.utils.num import brake, get_backend_module

pytest.importorskip("jax")
jnp = get_backend_module('jax')

from phaser.plan import SGDSolverPlan
from phaser.engines.gradient.solvers import SGDSolver


# ---- brake() ----------------------------------------------------------------

def test_brake_preserves_direction():
    rng = numpy.random.default_rng(0)
    v = jnp.array(rng.normal(0, 5, (20, 2)))
    result = numpy.asarray(brake(v, 1.0))
    v_np = numpy.asarray(v)

    # each row of `result` must be a non-negative scalar multiple of the
    # corresponding row of `v` (same direction, magnitude changed)
    cross = result[:, 0] * v_np[:, 1] - result[:, 1] * v_np[:, 0]
    numpy.testing.assert_allclose(cross, 0.0, atol=1e-8)
    dot = numpy.sum(result * v_np, axis=-1)
    assert numpy.all(dot >= 0)


def test_brake_never_exceeds_max():
    rng = numpy.random.default_rng(1)
    max_mag = 0.5
    v = jnp.array(rng.normal(0, 1000, (50, 2)))  # magnitudes far beyond the cap
    result = numpy.asarray(brake(v, max_mag))
    mags = numpy.linalg.norm(result, axis=-1)
    assert numpy.all(mags < max_mag)
    # and it should approach the cap closely for such large inputs
    assert numpy.all(mags > 0.9 * max_mag)


def test_brake_near_zero_is_near_identity():
    """For |v| << max_magnitude, arctan(x) ~= x, so brake(v, l) ~= v (no-op in the
    small-step regime, only the large-step regime gets meaningfully capped)."""
    max_mag = 10.0
    v = jnp.array([[1e-3, 2e-3], [-5e-4, 1e-4]])
    result = numpy.asarray(brake(v, max_mag))
    numpy.testing.assert_allclose(result, numpy.asarray(v), rtol=1e-4)


def test_brake_zero_input_is_zero():
    v = jnp.zeros((5, 2))
    result = numpy.asarray(brake(v, 1.0))
    numpy.testing.assert_array_equal(result, numpy.zeros((5, 2)))


def test_brake_monotonic_in_input_magnitude():
    """A larger raw update should never produce a *smaller* capped magnitude."""
    max_mag = 1.0
    dirs = jnp.array([[1.0, 0.0]])
    mags_in = [0.01, 0.1, 1.0, 10.0, 1000.0]
    out_mags = [float(jnp.linalg.norm(brake(dirs * m, max_mag))) for m in mags_in]
    assert all(a < b for (a, b) in zip(out_mags, out_mags[1:]))


# ---- SGDSolver max_step_size wiring -----------------------------------------

def _make_sgd_update(max_step_size, grad_positions, learning_rate=1.0, momentum=None):
    props = SGDSolverPlan(learning_rate=learning_rate, momentum=momentum, max_step_size=max_step_size)
    solver = SGDSolver({'plan': None, 'params': frozenset({'positions'})}, props)

    state = solver.init_state(sim=None)
    state = solver.update_for_iter(sim=None, state=state, niter=1)
    (update, _) = solver.update(sim=None, state=state, grad={'positions': grad_positions}, loss=0.0)
    return update['positions']


def test_sgd_max_step_size_caps_large_position_update():
    grad = jnp.array([[1000.0, 0.0], [0.0, -1000.0]])
    update = numpy.asarray(_make_sgd_update(max_step_size=0.05, grad_positions=grad, learning_rate=1.0))
    mags = numpy.linalg.norm(update, axis=-1)
    assert numpy.all(mags < 0.05)


def test_sgd_max_step_size_none_is_unaffected():
    """Default behavior (no cap set) must be bit-identical to before this feature existed."""
    grad = jnp.array([[3.0, -2.0], [0.5, 0.5]])
    capped_off = numpy.asarray(_make_sgd_update(max_step_size=None, grad_positions=grad, learning_rate=0.1))
    expected = numpy.asarray(grad) * 0.1
    numpy.testing.assert_allclose(capped_off, expected, rtol=1e-10)


def test_sgd_max_step_size_does_not_affect_object_grad():
    """max_step_size should only ever touch per-position vector variables
    (positions/tilt) -- an SGDSolver handling 'object' with max_step_size set
    must leave the object update completely untouched (no meaningful last-axis
    magnitude to cap)."""
    props = SGDSolverPlan(learning_rate=1.0, max_step_size=0.01)
    solver = SGDSolver({'plan': None, 'params': frozenset({'object'})}, props)

    state = solver.init_state(sim=None)
    state = solver.update_for_iter(sim=None, state=state, niter=1)

    obj_grad = jnp.ones((3, 8, 8), dtype=jnp.complex128) * (10.0 + 5.0j)
    (update, _) = solver.update(sim=None, state=state, grad={'object': obj_grad}, loss=0.0)

    numpy.testing.assert_allclose(numpy.asarray(update['object']), numpy.asarray(obj_grad), rtol=1e-10)
