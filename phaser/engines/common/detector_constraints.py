"""New per-iteration physical constraints ported from CuPy's `recon_12slices/ptycho/optimize.py`
(2026-09-15/16 revisions), for phaser's `background`/`propagator_mu` fields:

- `PropagatorMuClamp`: |exp(-mu)| can only ever remove intensity, never add it -- clip mu to a
  physically sane range every iteration, the same role CuPy's `mu_floor` clamp on `|p['p']|` plays.
- `NonNegBackground`: incoherent background is physically non-negative -- smooth floor at 0.
- `ProbeIntensityCap`/`BackgroundIntensityCap`: one-sided upper bounds on each field's total
  intensity (2026-09-16 CuPy revision -- only ever shrink when the ordinary gradient step
  overshoots the physical budget, never force up to match when under).

The intensity-cap constraints need two data-derived numbers `apply_iter(sim, state)` has no way to
reach on its own (no patterns, no per-iteration model totals): `M_bar` (mean measured intensity per
scan position, fixed for the whole run) and, for the background cap, the current iteration's
unmasked coherent total (`sim.progress['coh_total']`, populated by
`phaser.engines.gradient.run.run_model`/`run_engine` every iteration before constraints run). Rather
than changing the `IterConstraint` protocol for every constraint, `phaser.engines.gradient.run.run_engine`
calls an optional, duck-typed `bind_context(**kwargs)` once (after computing `M_bar`, right before
`iter_constraint_states` are initialized) on any constraint that defines it.
"""
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.state import ReconsState
from phaser.utils.num import get_array_module, abs2
from phaser.hooks.regularization import (
    PropagatorMuClampProps, NonNegBackgroundProps, ProbeIntensityCapProps, BackgroundIntensityCapProps,
)


class PropagatorMuClamp:
    def __init__(self, args: None, props: PropagatorMuClampProps):
        self.mu_max = props.mu_max

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        if sim.propagator_mu is not None:
            xp = get_array_module(sim.propagator_mu)
            sim.propagator_mu = xp.clip(sim.propagator_mu, 0.0, self.mu_max)
        return (sim, None)


class NonNegBackground:
    def __init__(self, args: None, props: NonNegBackgroundProps):
        self.eps = props.eps
        # bound lazily via bind_context; if never bound, the floor applies to every pixel
        # (harmless -- a pixel that never sees a nonzero gradient just also never moves off
        # whatever bind_context would have floored it towards anyway, since eps is tiny).
        self._ever_valid: t.Optional[NDArray[numpy.bool_]] = None

    def bind_context(self, *, pattern_mask: t.Optional[NDArray[numpy.floating]] = None, **_: t.Any) -> None:
        if pattern_mask is not None:
            self._ever_valid = pattern_mask > 0

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        if sim.background is None:
            return (sim, state)
        xp = get_array_module(sim.background)
        floored = 0.5 * (sim.background + xp.sqrt(sim.background**2 + self.eps**2))
        if self._ever_valid is not None:
            sim.background = xp.where(self._ever_valid, floored, sim.background)
        else:
            sim.background = floored
        return (sim, state)


class ProbeIntensityCap:
    """Upper-bounds sum(|probe|^2) at M_bar/npix (Parseval) -- the probe's total intensity should
    reflect only the coherent signal, so a smaller-than-naive total is physically fine; any
    shortfall belongs to the incoherent background instead. Only ever shrinks."""
    def __init__(self, args: None, props: ProbeIntensityCapProps):
        self._M_bar: t.Optional[t.Any] = None

    def bind_context(self, *, M_bar: t.Optional[t.Any] = None, **_: t.Any) -> None:
        if M_bar is not None:
            self._M_bar = M_bar

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        if self._M_bar is None:
            return (sim, state)
        xp = get_array_module(sim.probe.data)
        npix = sim.probe.data.shape[-2] * sim.probe.data.shape[-1]
        target = self._M_bar / npix
        cur = xp.sum(abs2(sim.probe.data))
        # xp.maximum(cur, eps) guards a near-zero probe: target/eps is then huge, so
        # minimum(1.0, ...) safely no-ops instead of dividing by zero.
        scale = xp.minimum(1.0, xp.sqrt(target / xp.maximum(cur, 1e-30)))
        sim.probe.data = (sim.probe.data * scale).astype(sim.probe.data.dtype)
        return (sim, state)


class BackgroundIntensityCap:
    """Upper-bounds the incoherent background's total intensity at `M_bar - coh_total/npos`
    (a leftover-power budget: what the coherent model hasn't already claimed anywhere in
    k-space, not just the valid/unmasked region -- multislice propagation conserves total
    coherent power, so restricting to the valid region alone wouldn't stop background inflating
    to cover power the object sent to masked pixels). Floored at 0; only ever shrinks."""
    def __init__(self, args: None, props: BackgroundIntensityCapProps):
        self._M_bar: t.Optional[t.Any] = None
        self._mask: t.Optional[NDArray[numpy.floating]] = None

    def bind_context(
        self, *, M_bar: t.Optional[t.Any] = None,
        pattern_mask: t.Optional[NDArray[numpy.floating]] = None, **_: t.Any,
    ) -> None:
        if M_bar is not None:
            self._M_bar = M_bar
        if pattern_mask is not None:
            self._mask = pattern_mask

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        if self._M_bar is None or sim.background is None:
            return (sim, state)
        xp = get_array_module(sim.background)
        coh_total_values = sim.progress['coh_total'].values if 'coh_total' in sim.progress else []
        # already a per-position average -- run_engine divides every progress value by
        # groups.n_pos before appending, same convention M_bar itself uses.
        coh_total_avg = coh_total_values[-1] if len(coh_total_values) else 0.0
        target_total = xp.maximum(self._M_bar - coh_total_avg, 0.0)
        cur_total = xp.sum(sim.background * self._mask) if self._mask is not None else xp.sum(sim.background)
        # xp.maximum(cur_total, eps) guards a near-zero background (e.g. before it's had any
        # gradient updates): target/eps is then huge, so minimum(1.0, ...) safely no-ops.
        scale = xp.minimum(1.0, target_total / xp.maximum(cur_total, 1e-30))
        sim.background = sim.background * scale
        return (sim, state)
