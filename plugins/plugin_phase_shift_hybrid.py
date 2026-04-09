from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request


class PolicyConfig:
    def __init__(
        self,
        init_probation_frac: float = 0.20,
        min_probation_frac: float = 0.08,
        max_probation_frac: float = 0.56,
        adapt_every: int = 1536,
        short_gap_epochs: int = 2,
        ghost_factor: int = 6,
        min_ghost_entries: int = 4096,
        max_ghost_entries: int = 500000,
    ):
        self.init_probation_frac = init_probation_frac
        self.min_probation_frac = min_probation_frac
        self.max_probation_frac = max_probation_frac
        self.adapt_every = adapt_every
        self.short_gap_epochs = short_gap_epochs
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries


class PhaseShiftHybrid:

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # probation: obj_id -> [size, refbit, last_epoch, ghost_mark]
        self.probation = OrderedDict()
        # protected: obj_id -> [size, refbit, last_epoch]
        self.protected = OrderedDict()
        # ghost: obj_id -> origin tag ('P' or 'R')
        self.ghost = OrderedDict()

        self.probation_bytes = 0
        self.protected_bytes = 0

        self.req_count = 0
        self.epoch = 0

        self.probation_target = self._clamp_probation_target(
            int(self.cache_size * self.cfg.init_probation_frac)
        )
        self.adapt_step = max(1, self.cache_size // 50)

        ghost_cap = self.cache_size * self.cfg.ghost_factor
        self.ghost_limit = max(
            self.cfg.min_ghost_entries, min(self.cfg.max_ghost_entries, ghost_cap)
        )

        # Short-window signals for fast adaptation.
        self.win_hit_probation = 0
        self.win_hit_protected = 0
        self.win_miss = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0
        self.win_promotions = 0

        # Smoothed phase signal and stress flags.
        self.phase_score = 0.0
        self.scan_mode = False
        self.protected_stress = False

    def _clamp_probation_target(self, target: int) -> int:
        lo = max(1, int(self.cache_size * self.cfg.min_probation_frac))
        hi = max(lo, int(self.cache_size * self.cfg.max_probation_frac))
        return min(hi, max(lo, target))

    def _on_request_tick(self) -> None:
        self.req_count += 1
        # Coarse epoch avoids expensive timestamps while keeping short-gap notion.
        if (self.req_count & 127) == 0:
            self.epoch += 1
        if self.req_count % self.cfg.adapt_every == 0:
            self._adapt()

    def _adapt(self) -> None:
        # Smoothed phase score: positive => reuse-protected phase, negative => scan/churn phase.
        window_score = (
            2 * self.win_hit_protected
            + self.win_hit_probation
            - 2 * self.win_miss
            - self.win_ghost_from_probation
        )
        self.phase_score = 0.7 * self.phase_score + 0.3 * float(window_score)

        # Fast scan pressure detection.
        hits = self.win_hit_probation + self.win_hit_protected
        self.scan_mode = (self.phase_score < -24.0) or (self.win_miss > hits + 64)

        delta = 0

        # If probation-evicted IDs come back, probation likely too small.
        if self.win_ghost_from_probation > self.win_ghost_from_protected + 2:
            delta += 1

        # If protected yields many hits, protect reuse by shrinking probation.
        if self.win_hit_protected > (self.win_hit_probation + self.win_ghost_from_probation + 8):
            delta -= 1

        # In churn/scan phases, give probation more room to absorb noise.
        if self.scan_mode:
            delta += 1

        protected_target = max(1, self.cache_size - self.probation_target)

        # Fallback stress rule: protected too big with poor yield => force more protected pressure.
        self.protected_stress = (
            self.protected_bytes > protected_target
            and self.win_hit_protected * 2 < self.win_miss
        )
        if self.protected_stress:
            delta += 1

        if delta != 0:
            self.probation_target = self._clamp_probation_target(
                self.probation_target + delta * self.adapt_step
            )

        # Reset short window counters.
        self.win_hit_probation = 0
        self.win_hit_protected = 0
        self.win_miss = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0
        self.win_promotions = 0

    def _ghost_add(self, obj_id: int, origin: str) -> None:
        self.ghost.pop(obj_id, None)
        self.ghost[obj_id] = origin
        if len(self.ghost) > self.ghost_limit:
            self.ghost.popitem(last=False)

    def _remove_obj(self, obj_id: int) -> None:
        rec = self.probation.pop(obj_id, None)
        if rec is not None:
            self.probation_bytes -= rec[0]
            return
        rec = self.protected.pop(obj_id, None)
        if rec is not None:
            self.protected_bytes -= rec[0]

    def _add_probation(
        self, obj_id: int, obj_size: int, refbit: int, ghost_mark: int, near_head: bool
    ) -> None:
        self._remove_obj(obj_id)
        self.probation[obj_id] = [obj_size, refbit, self.epoch, ghost_mark]
        if near_head:
            self.probation.move_to_end(obj_id, last=False)
        self.probation_bytes += obj_size

    def _add_protected(self, obj_id: int, obj_size: int, refbit: int, last_epoch: int) -> None:
        self._remove_obj(obj_id)
        self.protected[obj_id] = [obj_size, refbit, last_epoch]
        self.protected_bytes += obj_size

    def on_hit(self, req: Request) -> None:
        self._on_request_tick()
        obj_id = int(req.obj_id)

        rec = self.probation.get(obj_id)
        if rec is not None:
            self.win_hit_probation += 1
            obj_size, refbit, last_epoch, ghost_mark = rec
            short_gap = (self.epoch - last_epoch) <= self.cfg.short_gap_epochs

            # Conditional promotion:
            # - ghost-admitted entries promote on quick re-touch
            # - normal entries promote on quick re-touch outside scan mode
            if short_gap and (ghost_mark == 1 or not self.scan_mode):
                self.probation.pop(obj_id, None)
                self.probation_bytes -= obj_size
                self._add_protected(obj_id, obj_size, 1, self.epoch)
                self.win_promotions += 1
            else:
                rec[1] = 1
                rec[2] = self.epoch
                rec[3] = 0
            return

        rec = self.protected.get(obj_id)
        if rec is not None:
            self.win_hit_protected += 1
            rec[1] = 1
            rec[2] = self.epoch
            return

        # Defensive recovery for rare metadata desync.
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self._add_protected(obj_id, obj_size, 1, self.epoch)

    def on_miss(self, req: Request) -> None:
        self._on_request_tick()
        self.win_miss += 1

        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)
        if obj_size > self.cache_size:
            return

        self._remove_obj(obj_id)

        origin = self.ghost.pop(obj_id, None)
        if origin is not None:
            if origin == "P":
                self.win_ghost_from_probation += 1
            else:
                self.win_ghost_from_protected += 1
            # Important: ghost hits re-enter probation (not direct protected).
            self._add_probation(obj_id, obj_size, 1, 1, near_head=False)
            return

        # New entries go to probation; under scan pressure place near eviction edge.
        near_head = self.scan_mode and self.probation_bytes >= self.probation_target
        self._add_probation(obj_id, obj_size, 0, 0, near_head=near_head)

    def _evict_from_probation(self) -> Optional[int]:
        if not self.probation:
            return None

        n = len(self.probation)
        for _ in range(n):
            obj_id, rec = self.probation.popitem(last=False)
            obj_size, refbit, last_epoch, ghost_mark = rec
            self.probation_bytes -= obj_size

            if refbit == 0:
                self._ghost_add(obj_id, "P")
                return obj_id

            # Consistent second chance:
            # only recent/ghost-marked entries promote; others stay probation but cooled.
            short_gap = (self.epoch - last_epoch) <= self.cfg.short_gap_epochs
            if short_gap or ghost_mark == 1:
                self._add_protected(obj_id, obj_size, 0, last_epoch)
                self.win_promotions += 1
            else:
                self.probation[obj_id] = [obj_size, 0, last_epoch, 0]
                self.probation_bytes += obj_size

        # Fallback to guarantee progress.
        if self.probation:
            obj_id, rec = self.probation.popitem(last=False)
            self.probation_bytes -= rec[0]
            self._ghost_add(obj_id, "P")
            return obj_id

        return None

    def _evict_from_protected(self) -> Optional[int]:
        if not self.protected:
            return None

        n = len(self.protected)
        for _ in range(n):
            obj_id, rec = self.protected.popitem(last=False)
            obj_size, refbit, last_epoch = rec
            self.protected_bytes -= obj_size

            if refbit == 0:
                self._ghost_add(obj_id, "R")
                return obj_id

            # Second chance with ref reset.
            self.protected[obj_id] = [obj_size, 0, last_epoch]
            self.protected_bytes += obj_size

        # Fallback to guarantee progress.
        obj_id, rec = self.protected.popitem(last=False)
        self.protected_bytes -= rec[0]
        self._ghost_add(obj_id, "R")
        return obj_id

    def pick_victim(self, req: Request) -> int:
        _ = req

        protected_target = max(1, self.cache_size - self.probation_target)

        # Stress rule: if protected is bloated with poor yield, evict protected first.
        if self.protected_stress and self.protected and self.protected_bytes > protected_target:
            victim = self._evict_from_protected()
            if victim is not None:
                return victim

        # Default: prefer probation evictions for one-hit filtering.
        if self.probation and (self.probation_bytes > self.probation_target or not self.protected):
            victim = self._evict_from_probation()
            if victim is not None:
                return victim

        if self.probation:
            victim = self._evict_from_probation()
            if victim is not None:
                return victim

        victim = self._evict_from_protected()
        if victim is not None:
            return victim

        raise RuntimeError("eviction_hook called with empty policy metadata")

    def on_remove(self, obj_id: int) -> None:
        self._remove_obj(int(obj_id))

    def on_free(self) -> None:
        self.probation.clear()
        self.protected.clear()
        self.ghost.clear()
        self.probation_bytes = 0
        self.protected_bytes = 0


def init_hook(common_cache_params: CommonCacheParams) -> PhaseShiftHybrid:
    return PhaseShiftHybrid(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: PhaseShiftHybrid, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: PhaseShiftHybrid, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: PhaseShiftHybrid, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: PhaseShiftHybrid, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: PhaseShiftHybrid) -> None:
    data.on_free()
