from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request


class PolicyConfig:
    def __init__(
        self,
        init_probation_frac: float = 0.14,
        min_probation_frac: float = 0.05,
        max_probation_frac: float = 0.24,
        adapt_every: int = 512,
        short_gap_epochs: int = 3,
        ghost_factor: int = 4,
        min_ghost_entries: int = 2048,
        max_ghost_entries: int = 250000,
        adapt_step_div: int = 192,
        probation_scan_cap: int = 16,
        protected_scan_cap: int = 16,
        protected_overflow_slack_div: int = 50,
    ):
        self.init_probation_frac = init_probation_frac
        self.min_probation_frac = min_probation_frac
        self.max_probation_frac = max_probation_frac
        self.adapt_every = adapt_every
        self.short_gap_epochs = short_gap_epochs
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries
        self.adapt_step_div = adapt_step_div
        self.probation_scan_cap = probation_scan_cap
        self.protected_scan_cap = protected_scan_cap
        self.protected_overflow_slack_div = protected_overflow_slack_div


class BalancedGhostSieveV2:
    """
    Balanced scan-resistant online cache policy.

    Segments
    --------
    probation: new / uncertain objects
    protected: objects with convincing reuse
    ghost: remembers recent evictions and origin

    Main goals
    ----------
    - keep strong scan resistance
    - preserve more medium-gap reuse than harsher SIEVE variants
    - use gentle online adaptation
    - keep operations cheap and bounded
    """

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # probation: obj_id -> [size, hit_count, last_touch_epoch]
        self.probation = OrderedDict()
        # protected: obj_id -> [size, ref_bit]
        self.protected = OrderedDict()
        # ghost: obj_id -> [origin, ghost_epoch], origin in {"P", "R"}
        self.ghost = OrderedDict()

        self.probation_bytes = 0
        self.protected_bytes = 0

        self.req_count = 0
        self.epoch = 0

        self.probation_target = self._clamp_probation_target(
            int(self.cache_size * self.cfg.init_probation_frac)
        )
        self.adapt_step = max(1, self.cache_size // self.cfg.adapt_step_div)

        ghost_cap = self.cache_size * self.cfg.ghost_factor
        self.ghost_limit = max(
            self.cfg.min_ghost_entries,
            min(self.cfg.max_ghost_entries, ghost_cap),
        )

        # Rolling signals for gentle adaptation.
        self.win_hit_probation = 0
        self.win_hit_protected = 0
        self.win_miss = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0

        self.scan_mode = False

    def _clamp_probation_target(self, x: int) -> int:
        lo = max(1, int(self.cache_size * self.cfg.min_probation_frac))
        hi = max(lo, int(self.cache_size * self.cfg.max_probation_frac))
        return min(hi, max(lo, x))

    def _tick(self) -> None:
        self.req_count += 1
        if (self.req_count & 63) == 0:
            self.epoch += 1
        if self.req_count % self.cfg.adapt_every == 0:
            self._adapt()

    def _adapt(self) -> None:
        total_hits = self.win_hit_probation + self.win_hit_protected

        # Lightweight phase detector.
        self.scan_mode = self.win_miss > total_hits + 64

        delta = 0

        # If probation evictions keep coming back, probation is too small.
        if self.win_ghost_from_probation > self.win_hit_probation + 2:
            delta += 1

        # If protected evictions keep coming back, protected is too small.
        if self.win_ghost_from_protected > self.win_hit_protected + 2:
            delta -= 1

        # If protected is yielding many useful hits, preserve more of it.
        if self.win_hit_protected > self.win_ghost_from_probation + 8:
            delta -= 1

        # In scan-heavy phases, allow a slightly larger probation only if
        # protected is not clearly struggling.
        if self.scan_mode and self.win_ghost_from_protected == 0:
            delta += 1

        if delta != 0:
            self.probation_target = self._clamp_probation_target(
                self.probation_target + delta * self.adapt_step
            )

        self.win_hit_probation = 0
        self.win_hit_protected = 0
        self.win_miss = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0

    def _ghost_add(self, obj_id: int, origin: str) -> None:
        self.ghost.pop(obj_id, None)
        self.ghost[obj_id] = [origin, self.epoch]
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
        self,
        obj_id: int,
        obj_size: int,
        hit_count: int,
        last_touch_epoch: int,
    ) -> None:
        self._remove_obj(obj_id)
        self.probation[obj_id] = [obj_size, hit_count, last_touch_epoch]
        self.probation_bytes += obj_size

    def _add_protected(self, obj_id: int, obj_size: int, ref_bit: int) -> None:
        self._remove_obj(obj_id)
        self.protected[obj_id] = [obj_size, ref_bit]
        self.protected_bytes += obj_size

    def _demote_protected_overflow(self) -> None:
        """
        If protected grows too large, demote cold protected LRU objects into
        warm probation before fully evicting them. This preserves medium-gap
        reuse better than direct killing.
        """
        protected_target = max(1, self.cache_size - self.probation_target)
        slack = max(1, self.cache_size // self.cfg.protected_overflow_slack_div)

        while self.protected and self.protected_bytes > protected_target + slack:
            obj_id, rec = self.protected.popitem(last=False)
            obj_size, ref_bit = rec
            self.protected_bytes -= obj_size

            if ref_bit == 1:
                # Give one cheap second chance inside protected first.
                self.protected[obj_id] = [obj_size, 0]
                self.protected_bytes += obj_size
                break

            # Demote cold protected item into warm probation.
            self.probation[obj_id] = [obj_size, 1, self.epoch]
            self.probation_bytes += obj_size

    def on_hit(self, req: Request) -> None:
        self._tick()
        obj_id = int(req.obj_id)

        rec = self.probation.get(obj_id)
        if rec is not None:
            self.win_hit_probation += 1
            obj_size, hit_count, last_touch = rec
            short_gap = (self.epoch - last_touch) <= self.cfg.short_gap_epochs

            # Promote on clear evidence only:
            # - quick reuse, or
            # - second real probation hit
            if short_gap or hit_count >= 1:
                self.probation.pop(obj_id, None)
                self.probation_bytes -= obj_size
                self._add_protected(obj_id, obj_size, 1)
            else:
                rec[1] = min(2, hit_count + 1)
                rec[2] = self.epoch
            return

        rec = self.protected.get(obj_id)
        if rec is not None:
            self.win_hit_protected += 1
            rec[1] = 1
            self.protected.move_to_end(obj_id, last=True)
            return

        # Defensive recovery: keep it cautious, not direct-protected.
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self._add_probation(obj_id, obj_size, 1, self.epoch)

    def on_miss(self, req: Request) -> None:
        self._tick()
        self.win_miss += 1

        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        if obj_size > self.cache_size:
            return

        self._remove_obj(obj_id)

        ghost_rec = self.ghost.pop(obj_id, None)
        if ghost_rec is not None:
            origin, ghost_epoch = ghost_rec
            recent_ghost = (self.epoch - ghost_epoch) <= self.cfg.short_gap_epochs

            if origin == "P":
                self.win_ghost_from_probation += 1
                # Probation comeback is warm but cautious.
                self._add_probation(obj_id, obj_size, 1, self.epoch)
                return

            # origin == "R"
            self.win_ghost_from_protected += 1
            # More generous protected comeback rule than v1:
            # recent protected ghost hits, or clear protected usefulness,
            # come straight back to protected.
            if recent_ghost or (
                not self.scan_mode and
                self.win_hit_protected >= self.win_hit_probation
            ):
                self._add_protected(obj_id, obj_size, 1)
            else:
                self._add_probation(obj_id, obj_size, 1, self.epoch)
            return

        # First touch enters cold probation.
        self._add_probation(obj_id, obj_size, 0, self.epoch)

    def _evict_from_probation(self):
        if not self.probation:
            return None

        scan_cap = min(len(self.probation), self.cfg.probation_scan_cap)

        for _ in range(scan_cap):
            obj_id, rec = self.probation.popitem(last=False)
            obj_size, hit_count, last_touch = rec
            self.probation_bytes -= obj_size

            # Cold probationary item: best eviction target.
            if hit_count <= 0:
                self._ghost_add(obj_id, "P")
                return obj_id

            short_gap = (self.epoch - last_touch) <= self.cfg.short_gap_epochs

            # Warm + recent gets protected if not in scan-heavy phase.
            if short_gap and not self.scan_mode:
                self._add_protected(obj_id, obj_size, 0)
                continue

            # Otherwise one softer extra probation lap.
            self.probation[obj_id] = [obj_size, hit_count - 1, last_touch]
            self.probation_bytes += obj_size

        # Guaranteed progress after capped scan.
        if self.probation:
            obj_id, rec = self.probation.popitem(last=False)
            self.probation_bytes -= rec[0]
            self._ghost_add(obj_id, "P")
            return obj_id

        return None

    def _evict_from_protected(self):
        if not self.protected:
            return None

        scan_cap = min(len(self.protected), self.cfg.protected_scan_cap)

        for _ in range(scan_cap):
            obj_id, rec = self.protected.popitem(last=False)
            obj_size, ref_bit = rec
            self.protected_bytes -= obj_size

            if ref_bit == 0:
                self._ghost_add(obj_id, "R")
                return obj_id

            # Cheap second chance.
            self.protected[obj_id] = [obj_size, 0]
            self.protected_bytes += obj_size

        # Guaranteed progress after capped scan.
        if self.protected:
            obj_id, rec = self.protected.popitem(last=False)
            self.protected_bytes -= rec[0]
            self._ghost_add(obj_id, "R")
            return obj_id

        return None

    def pick_victim(self, req: Request) -> int:
        _ = req

        # First calm protected overflow by demotion, not immediate killing.
        self._demote_protected_overflow()

        protected_target = max(1, self.cache_size - self.probation_target)

        # If protected is still oversized, trim it.
        if self.protected and self.protected_bytes > protected_target:
            victim = self._evict_from_protected()
            if victim is not None:
                return victim

        # Otherwise clean probation first for scan resistance.
        if self.probation and (
            self.probation_bytes > self.probation_target or not self.protected
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


def init_hook(common_cache_params: CommonCacheParams) -> BalancedGhostSieveV2:
    return BalancedGhostSieveV2(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: BalancedGhostSieveV2, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: BalancedGhostSieveV2, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: BalancedGhostSieveV2, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: BalancedGhostSieveV2, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: BalancedGhostSieveV2) -> None:
    data.on_free()