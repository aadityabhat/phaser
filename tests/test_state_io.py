"""HDF5 round-trip coverage for the new `background`/`propagator_mu` optional `ReconsState`
fields (`phaser.utils.io.hdf5_write_state`/`hdf5_read_state`), mirroring the existing `tilt`
field's optional-dataset pattern. Also covers `ObjectSampling.region_min`/`region_max`'s
None case (`_hdf5_read_nullable_dataset_shape`), a real pre-existing crash surfaced while
writing this file's original version -- see `test_object_sampling_region_min_max_roundtrip_when_none`.
"""
import io

import numpy
import h5py

from phaser.utils.num import Sampling
from phaser.utils.object import ObjectSampling
from phaser.state import ObjectState, ProbeState, ReconsState, IterState


def _make_state(background=None, propagator_mu=None, region_min=None, region_max=None) -> ReconsState:
    rng = numpy.random.default_rng(0)
    (ny, nx) = (12, 12)
    (by, bx) = (6, 6)

    obj_sampling = ObjectSampling((ny, nx), sampling=(1.0, 1.0), region_min=region_min, region_max=region_max)
    obj_data = (rng.normal(0, 0.3, (1, ny, nx)) + 1j * rng.normal(0, 0.3, (1, ny, nx))).astype(numpy.complex128)
    obj_state = ObjectState(obj_sampling, obj_data, numpy.array([1.0]))

    probe_sampling = Sampling((by, bx), sampling=(1.0, 1.0))
    probe_data = (rng.normal(0, 1, (1, by, bx)) + 1j * rng.normal(0, 1, (1, by, bx))).astype(numpy.complex128)
    probe_state = ProbeState(probe_sampling, probe_data)

    scan = rng.uniform(-4, 4, (5, 2))

    return ReconsState(
        iter=IterState.empty(), wavelength=1.0, probe=probe_state, object=obj_state,
        scan=scan, tilt=None, background=background, propagator_mu=propagator_mu, progress={},
    )


def _roundtrip(state: ReconsState) -> ReconsState:
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        state.write_hdf5(f)
    buf.seek(0)
    with h5py.File(buf, 'r') as f:
        return ReconsState.read_hdf5(f)


def test_background_and_propagator_mu_roundtrip_when_present():
    background = numpy.linspace(0, 1, 36).reshape(6, 6)
    propagator_mu = numpy.linspace(-1, 1, 36).reshape(6, 6)
    state = _make_state(background=background, propagator_mu=propagator_mu)

    read_back = _roundtrip(state)

    assert read_back.background is not None
    assert read_back.propagator_mu is not None
    numpy.testing.assert_allclose(read_back.background, background)
    numpy.testing.assert_allclose(read_back.propagator_mu, propagator_mu)


def test_background_and_propagator_mu_absent_when_none():
    """A checkpoint saved before these fields existed (or a run that never turned them
    on) must still round-trip cleanly, with both fields coming back None -- not a
    missing-key crash."""
    state = _make_state(background=None, propagator_mu=None)

    read_back = _roundtrip(state)

    assert read_back.background is None
    assert read_back.propagator_mu is None


def test_to_xp_and_to_numpy_preserve_background_and_propagator_mu():
    background = numpy.linspace(0, 1, 36).reshape(6, 6)
    propagator_mu = numpy.linspace(-1, 1, 36).reshape(6, 6)
    state = _make_state(background=background, propagator_mu=propagator_mu)

    roundtripped = state.to_xp(numpy).to_numpy()

    numpy.testing.assert_allclose(roundtripped.background, background)
    numpy.testing.assert_allclose(roundtripped.propagator_mu, propagator_mu)


def test_object_sampling_region_min_max_roundtrip_when_none():
    """Regression test: `_hdf5_write_nullable_dataset` writes an `h5py.Empty` placeholder
    for a None `region_min`/`region_max` (the default -- most `ObjectSampling`s never set
    these), and the read side used to crash on it (`AttributeError: 'Empty' object has no
    attribute 'astype'`, from treating the Empty placeholder as an ordinary dataset). Fixed
    by `_hdf5_read_nullable_dataset_shape` explicitly checking for it
    (`dataset.shape is None`) before delegating to the ordinary read path."""
    state = _make_state(region_min=None, region_max=None)
    assert state.object.sampling.region_min is None
    assert state.object.sampling.region_max is None

    read_back = _roundtrip(state)

    assert read_back.object.sampling.region_min is None
    assert read_back.object.sampling.region_max is None


def test_object_sampling_region_min_max_roundtrip_when_present():
    state = _make_state(region_min=(1.0, 2.0), region_max=(10.0, 11.0))

    read_back = _roundtrip(state)

    numpy.testing.assert_allclose(read_back.object.sampling.region_min, [1.0, 2.0])
    numpy.testing.assert_allclose(read_back.object.sampling.region_max, [10.0, 11.0])
