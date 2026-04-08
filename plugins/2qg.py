from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request

class PolicyConfig:
    def __init__(
        self,
        window_fraction: float = 0.20,
        min_window_bytes: int = 64 * 1024,
        ghost_factor: int = 4,
        min_ghost_entries: int = 10_000,
        max_ghost_entries: int = 400_000,
    ):
        self.window_fraction = window_fraction
        self.min_window_bytes = min_window_bytes
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries


class Competition2QGhost:
    """Byte-aware 2Q + ghost directory policy metadata."""

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # OrderedDict order is oldest -> newest.
        self.window: "OrderedDict[int, int]" = OrderedDict()
        self.main: "OrderedDict[int, int]" = OrderedDict()

        # Recently evicted IDs only (ghost metadata, no object bytes).
        self.ghost: "OrderedDict[int, None]" = OrderedDict()

        self.window_bytes = 0
        self.main_bytes = 0

        self.avg_obj_size = 1024.0
        self.window_target = self._compute_window_target()
        self.main_target = max(0, self.cache_size - self.window_target)
        self.ghost_limit = self._compute_ghost_limit()

    def _compute_window_target(self) -> int:
        return min(
            self.cache_size,
            max(self.cfg.min_window_bytes, int(self.cache_size * self.cfg.window_fraction)),
        )

    def _compute_ghost_limit(self) -> int:
        est_objects = max(1, int(self.cache_size / max(1.0, self.avg_obj_size)))
        limit = est_objects * self.cfg.ghost_factor
        return max(self.cfg.min_ghost_entries, min(self.cfg.max_ghost_entries, limit))

    def _update_size_ewma(self, obj_size: int) -> None:
        alpha = 0.01
        self.avg_obj_size = (1.0 - alpha) * self.avg_obj_size + alpha * float(max(1, obj_size))
        self.ghost_limit = self._compute_ghost_limit()

    def _ghost_add(self, obj_id: int) -> None:
        if obj_id in self.ghost:
            self.ghost.move_to_end(obj_id, last=True)
            return
        self.ghost[obj_id] = None
        if len(self.ghost) > self.ghost_limit:
            self.ghost.popitem(last=False)

    def _remove_obj(self, obj_id: int) -> None:
        size = self.window.pop(obj_id, None)
        if size is not None:
            self.window_bytes -= size
            return

        size = self.main.pop(obj_id, None)
        if size is not None:
            self.main_bytes -= size

    def _add_window(self, obj_id: int, obj_size: int) -> None:
        self.window[obj_id] = obj_size
        self.window.move_to_end(obj_id, last=True)
        self.window_bytes += obj_size

    def _add_main(self, obj_id: int, obj_size: int) -> None:
        self.main[obj_id] = obj_size
        self.main.move_to_end(obj_id, last=True)
        self.main_bytes += obj_size

    def _rebalance_segments(self) -> None:
        # Keep Main near its target; demote oldest Main objects to Window metadata.
        while self.main and self.main_bytes > self.main_target:
            oid, osz = self.main.popitem(last=False)
            self.main_bytes -= osz
            self._add_window(oid, osz)

    def on_hit(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)
        self._update_size_ewma(obj_size)

        if obj_id in self.main:
            # Main behaves like protected LRU.
            self.main.move_to_end(obj_id, last=True)
            return

        if obj_id in self.window:
            # Second hit: promote out of probation to protected segment.
            osz = self.window.pop(obj_id)
            self.window_bytes -= osz
            self._add_main(obj_id, osz)
            self._rebalance_segments()
            return

        # Rare mismatch recovery: if metadata is missing, treat as hot insert.
        if obj_size <= self.cache_size:
            self._add_main(obj_id, obj_size)
            self._rebalance_segments()

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)
        self._update_size_ewma(obj_size)

        # Oversized objects are not inserted by core cache; do not track them.
        if obj_size > self.cache_size:
            return

        # Defensive dedupe in case metadata got stale.
        self._remove_obj(obj_id)

        if obj_id in self.ghost:
            # Re-reference after recent eviction: bypass probation.
            self.ghost.pop(obj_id, None)
            self._add_main(obj_id, obj_size)
        else:
            # First admission goes to probation window.
            self._add_window(obj_id, obj_size)

        self._rebalance_segments()

    def pick_victim(self, req: Request) -> int:
        # Eviction logic:
        # 1) Prefer evicting oldest probation entries (filters one-hit objects).
        # 2) Fall back to protected segment when probation is empty.
        if self.window:
            victim_id, victim_size = self.window.popitem(last=False)
            self.window_bytes -= victim_size
            self._ghost_add(victim_id)
            return victim_id

        if self.main:
            victim_id, victim_size = self.main.popitem(last=False)
            self.main_bytes -= victim_size
            self._ghost_add(victim_id)
            return victim_id

        # Should never happen in steady state; keep it safe for debugging.
        raise RuntimeError(
            "eviction_hook called with empty policy metadata; cannot choose victim"
        )

    def on_remove(self, obj_id: int) -> None:
        # Must be tolerant: remove may be called for IDs we do not track.
        self._remove_obj(int(obj_id))

    def on_free(self) -> None:
        self.window.clear()
        self.main.clear()
        self.ghost.clear()
        self.window_bytes = 0
        self.main_bytes = 0

def init_hook(common_cache_params: CommonCacheParams) -> Competition2QGhost:
    return Competition2QGhost(cache_size=int(common_cache_params.cache_size))

def hit_hook(data: Competition2QGhost, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: Competition2QGhost, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: Competition2QGhost, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: Competition2QGhost, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: Competition2QGhost) -> None:
    data.on_free()
