from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request


class PolicyConfig:
    def __init__(
        self,
        init_small_frac: float = 0.10,
        min_small_frac: float = 0.05,
        max_small_frac: float = 0.50,
        min_small_bytes: int = 16,
        ghost_factor: int = 4,
        min_ghost_entries: int = 2048,
        max_ghost_entries: int = 400000,
        max_freq: int = 3,
    ):
        self.init_small_frac = init_small_frac
        self.min_small_frac = min_small_frac
        self.max_small_frac = max_small_frac
        self.min_small_bytes = min_small_bytes
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries
        self.max_freq = max_freq


class AdaptiveS3FIFO:

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # obj_id -> [obj_size, freq]
        self.small = OrderedDict()
        self.main = OrderedDict()
        # ghost: obj_id -> origin segment ('S' or 'M')
        self.ghost = OrderedDict()

        self.small_bytes = 0
        self.main_bytes = 0

        # Adaptive target for probation bytes.
        self.min_small_target = min(
            self.cache_size,
            max(self.cfg.min_small_bytes, int(self.cache_size * self.cfg.min_small_frac)),
        )
        self.max_small_target = min(
            self.cache_size,
            max(self.cfg.min_small_bytes, int(self.cache_size * self.cfg.max_small_frac)),
        )
        self.small_target = min(
            self.max_small_target,
            max(
                self.min_small_target,
                int(self.cache_size * self.cfg.init_small_frac),
            ),
        )
        self.adapt_step = max(1, self.cache_size // 100)

        est_objects = max(1, self.cache_size)
        ghost_cap = est_objects * self.cfg.ghost_factor
        self.ghost_limit = max(
            self.cfg.min_ghost_entries, min(self.cfg.max_ghost_entries, ghost_cap)
        )

    def _add_ghost(self, obj_id: int, origin: str) -> None:
        old = self.ghost.pop(obj_id, None)
        _ = old
        self.ghost[obj_id] = origin
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

    def _add_small(self, obj_id: int, obj_size: int, freq: int) -> None:
        old = self.small.pop(obj_id, None)
        if old is not None:
            self.small_bytes -= old[0]
        old = self.main.pop(obj_id, None)
        if old is not None:
            self.main_bytes -= old[0]

        self.small[obj_id] = [obj_size, freq]
        self.small_bytes += obj_size

    def _add_main(self, obj_id: int, obj_size: int, freq: int) -> None:
        old = self.main.pop(obj_id, None)
        if old is not None:
            self.main_bytes -= old[0]
        old = self.small.pop(obj_id, None)
        if old is not None:
            self.small_bytes -= old[0]

        self.main[obj_id] = [obj_size, freq]
        self.main_bytes += obj_size

    def _adapt_on_ghost_hit(self, origin: str) -> None:
        # If items evicted from small are hit again, grow small.
        # If items evicted from main are hit again, shrink small.
        if origin == "S":
            self.small_target = min(self.max_small_target, self.small_target + self.adapt_step)
        elif origin == "M":
            self.small_target = max(self.min_small_target, self.small_target - self.adapt_step)

    def on_hit(self, req: Request) -> None:
        obj_id = int(req.obj_id)

        rec = self.small.get(obj_id)
        if rec is not None:
            if rec[1] < self.cfg.max_freq:
                rec[1] += 1
            return

        rec = self.main.get(obj_id)
        if rec is not None:
            if rec[1] < self.cfg.max_freq:
                rec[1] += 1
            return

        # Rare metadata desync fallback.
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self._add_main(obj_id, obj_size, 1)

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        # Core cache won't insert oversized objects; don't track them.
        if obj_size > self.cache_size:
            return

        self._remove_obj(obj_id)

        origin = self.ghost.pop(obj_id, None)
        if origin is not None:
            self._adapt_on_ghost_hit(origin)
            # Fast-track likely useful re-references to main.
            self._add_main(obj_id, obj_size, 1)
        else:
            # New objects start probationary and cold.
            self._add_small(obj_id, obj_size, 0)

    def _evict_from_small(self) -> Optional[int]:
        if not self.small:
            return None

        # Walk probation FIFO once:
        # - cold item: evict now
        # - warm item: decay freq and promote to main
        n = len(self.small)
        for _ in range(n):
            obj_id, rec = self.small.popitem(last=False)
            obj_size, freq = rec
            self.small_bytes -= obj_size

            if freq <= 0:
                self._add_ghost(obj_id, "S")
                return obj_id

            self._add_main(obj_id, obj_size, freq - 1)

        return None

    def _evict_from_main(self) -> Optional[int]:
        if not self.main:
            return None

        # Reinsertion with frequency decay.
        n = len(self.main)
        for _ in range(n):
            obj_id, rec = self.main.popitem(last=False)
            obj_size, freq = rec
            self.main_bytes -= obj_size

            if freq <= 0:
                self._add_ghost(obj_id, "M")
                return obj_id

            self.main[obj_id] = [obj_size, freq - 1]
            self.main_bytes += obj_size

        # Guarantee progress if all entries were warm.
        obj_id, rec = self.main.popitem(last=False)
        self.main_bytes -= rec[0]
        self._add_ghost(obj_id, "M")
        return obj_id

    def pick_victim(self, req: Request) -> int:
        _ = req

        # Prefer evicting from small when it exceeds its target or when main is empty.
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


def init_hook(common_cache_params: CommonCacheParams) -> AdaptiveS3FIFO:
    return AdaptiveS3FIFO(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: AdaptiveS3FIFO, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: AdaptiveS3FIFO, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: AdaptiveS3FIFO, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: AdaptiveS3FIFO, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: AdaptiveS3FIFO) -> None:
    data.on_free()
