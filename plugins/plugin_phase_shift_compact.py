from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request


class PolicyConfig:
    def __init__(
        self,
        init_probation_frac: float = 0.24,
        min_probation_frac: float = 0.10,
        max_probation_frac: float = 0.60,
        adapt_every: int = 2048,
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


class PhaseShiftCompact:
    """
    Compact adaptive FIFO hybrid:
    - probation FIFO for new objects
    - protected FIFO for proven reuse
    - ghost FIFO for adaptation feedback

    Design choices:
    - promotion only on probation hits (not on eviction scans)
    - ghost hits re-enter probation with a warm bit, not direct protected admission
    - adaptation uses one primary outcome signal:
      ghost-from-probation pressure vs protected-hit yield
    """

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # probation: obj_id -> [size, refbit, last_epoch]
        self.probation = OrderedDict()
        # protected: obj_id -> [size, refbit]
        self.protected = OrderedDict()
        # ghost: obj_id -> origin ('P' or 'R')
        self.ghost = OrderedDict()

        self.probation_bytes = 0
        self.protected_bytes = 0

        self.req_count = 0
        self.epoch = 0

        self.probation_target = self._clamp_probation_target(
            int(self.cache_size * self.cfg.init_probation_frac)
        )
        self.adapt_step = max(1, self.cache_size // 40)

        ghost_cap = self.cache_size * self.cfg.ghost_factor
        self.ghost_limit = max(
            self.cfg.min_ghost_entries, min(self.cfg.max_ghost_entries, ghost_cap)
        )

        # Short rolling signals.
        self.win_hit_protected = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0

    def _clamp_probation_target(self, x: int) -> int:
        lo = max(1, int(self.cache_size * self.cfg.min_probation_frac))
        hi = max(lo, int(self.cache_size * self.cfg.max_probation_frac))
        return min(hi, max(lo, x))

    def _tick(self) -> None:
        self.req_count += 1
        if (self.req_count & 127) == 0:
            self.epoch += 1
        if self.req_count % self.cfg.adapt_every == 0:
            self._adapt()

    def _adapt(self) -> None:
        delta = 0
        # If many probation evictees come back, probation is too small.
        if self.win_ghost_from_probation > self.win_hit_protected + 2:
            delta += 1
        # If protected is producing strong hits, give it more room.
        if self.win_hit_protected > self.win_ghost_from_probation + 6:
            delta -= 1
        # If ghost says protected evictions also come back, bias to larger probation.
        if self.win_ghost_from_protected > self.win_hit_protected + 4:
            delta += 1

        if delta != 0:
            self.probation_target = self._clamp_probation_target(
                self.probation_target + delta * self.adapt_step
            )

        self.win_hit_protected = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0

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

    def _add_probation(self, obj_id: int, obj_size: int, refbit: int) -> None:
        self._remove_obj(obj_id)
        self.probation[obj_id] = [obj_size, refbit, self.epoch]
        self.probation_bytes += obj_size

    def _add_protected(self, obj_id: int, obj_size: int, refbit: int) -> None:
        self._remove_obj(obj_id)
        self.protected[obj_id] = [obj_size, refbit]
        self.protected_bytes += obj_size

    def on_hit(self, req: Request) -> None:
        self._tick()
        obj_id = int(req.obj_id)

        rec = self.probation.get(obj_id)
        if rec is not None:
            obj_size, refbit, last_epoch = rec
            short_gap = (self.epoch - last_epoch) <= self.cfg.short_gap_epochs
            # Promote only on clear evidence of quick reuse.
            if short_gap or refbit == 1:
                self.probation.pop(obj_id, None)
                self.probation_bytes -= obj_size
                self._add_protected(obj_id, obj_size, 1)
            else:
                rec[1] = 1
                rec[2] = self.epoch
            return

        rec = self.protected.get(obj_id)
        if rec is not None:
            self.win_hit_protected += 1
            rec[1] = 1
            return

        # Defensive metadata recovery.
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self._add_protected(obj_id, obj_size, 1)

    def on_miss(self, req: Request) -> None:
        self._tick()
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
            # Conditional ghost handling: warm probation, not direct protected.
            self._add_probation(obj_id, obj_size, 1)
            return

        # First touch enters probation cold.
        self._add_probation(obj_id, obj_size, 0)

    def _evict_from_probation(self) -> Optional[int]:
        if not self.probation:
            return None

        n = len(self.probation)
        for _ in range(n):
            obj_id, rec = self.probation.popitem(last=False)
            obj_size, refbit, _last_epoch = rec
            self.probation_bytes -= obj_size

            if refbit == 0:
                self._ghost_add(obj_id, "P")
                return obj_id

            # Quick demotion bias: give exactly one extra turn in probation.
            self.probation[obj_id] = [obj_size, 0, self.epoch]
            self.probation_bytes += obj_size

        # Guaranteed progress.
        obj_id, rec = self.probation.popitem(last=False)
        self.probation_bytes -= rec[0]
        self._ghost_add(obj_id, "P")
        return obj_id

    def _evict_from_protected(self) -> Optional[int]:
        if not self.protected:
            return None

        n = len(self.protected)
        for _ in range(n):
            obj_id, rec = self.protected.popitem(last=False)
            obj_size, refbit = rec
            self.protected_bytes -= obj_size

            if refbit == 0:
                self._ghost_add(obj_id, "R")
                return obj_id

            # One second chance, then become cold.
            self.protected[obj_id] = [obj_size, 0]
            self.protected_bytes += obj_size

        # Guaranteed progress.
        obj_id, rec = self.protected.popitem(last=False)
        self.protected_bytes -= rec[0]
        self._ghost_add(obj_id, "R")
        return obj_id

    def pick_victim(self, req: Request) -> int:
        _ = req

        # Hard bias toward quick probation cleanup for scan resistance.
        if self.probation and (
            self.probation_bytes > self.probation_target or self.protected_bytes == 0
        ):
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


def init_hook(common_cache_params: CommonCacheParams) -> PhaseShiftCompact:
    return PhaseShiftCompact(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: PhaseShiftCompact, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: PhaseShiftCompact, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: PhaseShiftCompact, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: PhaseShiftCompact, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: PhaseShiftCompact) -> None:
    data.on_free()
