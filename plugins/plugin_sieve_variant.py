from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request


class PolicyConfig:
    def __init__(
        self,
        small_fraction: float = 0.20,
        min_small_bytes: int = 16,
        ghost_factor: int = 4,
        min_ghost_entries: int = 2048,
        max_ghost_entries: int = 400000,
    ):
        self.small_fraction = small_fraction
        self.min_small_bytes = min_small_bytes
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries


class SieveVariantCache:
    """
    SIEVE-inspired selective FIFO:
    - New objects enter a small probationary FIFO (small queue).
    - Hits only set a reference bit (no expensive reordering).
    - Eviction prefers small-queue cold entries (one-hit filtering).
    - Referenced small entries are promoted to main queue.
    - Ghost hits bypass probation and go directly to main.
    """

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # obj_id -> [obj_size, refbit]; queue order oldest -> newest
        self.small = OrderedDict()
        self.main = OrderedDict()
        self.ghost = OrderedDict()  # obj_id -> None

        self.small_bytes = 0
        self.main_bytes = 0

        self.small_target = min(
            self.cache_size,
            max(self.cfg.min_small_bytes, int(self.cache_size * self.cfg.small_fraction)),
        )
        est_objects = max(1, self.cache_size)
        ghost_cap = est_objects * self.cfg.ghost_factor
        self.ghost_limit = max(
            self.cfg.min_ghost_entries, min(self.cfg.max_ghost_entries, ghost_cap)
        )

    def _ghost_add(self, obj_id: int) -> None:
        if obj_id in self.ghost:
            self.ghost.move_to_end(obj_id, last=True)
            return
        self.ghost[obj_id] = None
        if len(self.ghost) > self.ghost_limit:
            self.ghost.popitem(last=False)

    def _remove_obj(self, obj_id: int) -> None:
        rec = self.small.pop(obj_id, None)
        if rec is not None:
            self.small_bytes -= rec[0]
            return
        rec = self.main.pop(obj_id, None)
        if rec is not None:
            self.main_bytes -= rec[0]

    def _add_small(self, obj_id: int, obj_size: int, refbit: int) -> None:
        old = self.small.pop(obj_id, None)
        if old is not None:
            self.small_bytes -= old[0]
        old = self.main.pop(obj_id, None)
        if old is not None:
            self.main_bytes -= old[0]
        self.small[obj_id] = [obj_size, refbit]
        self.small_bytes += obj_size

    def _add_main(self, obj_id: int, obj_size: int, refbit: int) -> None:
        old = self.main.pop(obj_id, None)
        if old is not None:
            self.main_bytes -= old[0]
        old = self.small.pop(obj_id, None)
        if old is not None:
            self.small_bytes -= old[0]
        self.main[obj_id] = [obj_size, refbit]
        self.main_bytes += obj_size

    def on_hit(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        rec = self.small.get(obj_id)
        if rec is not None:
            rec[1] = 1
            return
        rec = self.main.get(obj_id)
        if rec is not None:
            rec[1] = 1
            return

        # Defensive fallback for rare metadata desync.
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self._add_main(obj_id, obj_size, 1)

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        # Oversized requests are not inserted by core cache.
        if obj_size > self.cache_size:
            return

        self._remove_obj(obj_id)

        if obj_id in self.ghost:
            self.ghost.pop(obj_id, None)
            # Re-reference after eviction: treat as likely hot.
            self._add_main(obj_id, obj_size, 1)
        else:
            # First admission is intentionally cold/probationary.
            self._add_small(obj_id, obj_size, 0)

    def _evict_from_small(self) -> Optional[int]:
        if not self.small:
            return None

        # One pass over probation queue:
        # - cold objects are evicted immediately,
        # - referenced ones are promoted to main.
        n = len(self.small)
        for _ in range(n):
            obj_id, rec = self.small.popitem(last=False)
            obj_size, refbit = rec
            self.small_bytes -= obj_size

            if refbit:
                self._add_main(obj_id, obj_size, 0)
                continue

            self._ghost_add(obj_id)
            return obj_id

        return None

    def _evict_from_main(self) -> Optional[int]:
        if not self.main:
            return None

        # CLOCK-like second chance in FIFO order.
        n = len(self.main)
        for _ in range(n):
            obj_id, rec = self.main.popitem(last=False)
            obj_size, refbit = rec
            self.main_bytes -= obj_size

            if refbit:
                self.main[obj_id] = [obj_size, 0]
                self.main_bytes += obj_size
                continue

            self._ghost_add(obj_id)
            return obj_id

        # If everything had refbit=1, one more pop guarantees progress.
        obj_id, rec = self.main.popitem(last=False)
        self.main_bytes -= rec[0]
        self._ghost_add(obj_id)
        return obj_id

    def pick_victim(self, req: Request) -> int:
        _ = req

        # Keep small queue near target and prioritize filtering one-hit objects.
        if self.small and (self.small_bytes > self.small_target or not self.main):
            victim = self._evict_from_small()
            if victim is not None:
                return victim

        victim = self._evict_from_small()
        if victim is not None:
            return victim

        victim = self._evict_from_main()
        if victim is not None:
            return victim

        raise RuntimeError("eviction_hook called with empty policy metadata")

    def on_remove(self, obj_id: int) -> None:
        # Must tolerate unknown IDs.
        self._remove_obj(int(obj_id))

    def on_free(self) -> None:
        self.small.clear()
        self.main.clear()
        self.ghost.clear()
        self.small_bytes = 0
        self.main_bytes = 0


def init_hook(common_cache_params: CommonCacheParams) -> SieveVariantCache:
    return SieveVariantCache(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: SieveVariantCache, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: SieveVariantCache, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: SieveVariantCache, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: SieveVariantCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: SieveVariantCache) -> None:
    data.on_free()
