from collections import OrderedDict
from heapq import heappop, heappush
from typing import Optional

from libcachesim import CommonCacheParams, Request

INF_NEXT = 1 << 62

class PolicyConfig:
    def __init__(
        self,
        small_fraction: float = 0.10,
        min_small_bytes: int = 16,
        ghost_factor: int = 4,
        min_ghost_entries: int = 2048,
        max_ghost_entries: int = 400000,
        oracle_enable_threshold: int = 32,
    ):
        self.small_fraction = small_fraction
        self.min_small_bytes = min_small_bytes
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries
        self.oracle_enable_threshold = oracle_enable_threshold


class OracleBeladyHybrid:
    """
    - If next_access_vtime is available, use Belady-like eviction via max-next heap. Otherwise use SIEVE hybrid type of stuff.
    """

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # obj_id -> [size, refbit, next_v, stamp]
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
        self.oracle_mode = False
        self.oracle_nonneg_seen = 0
        self.req_seen = 0
        self.stamp = 0

        # Heap entries: (-next_v, stamp, obj_id)
        self.max_next_heap = []

    def _next_from_req(self, req: Request) -> int:

        if (
            not self.oracle_mode
            and self.oracle_nonneg_seen >= self.cfg.oracle_enable_threshold
    ):
            self.oracle_mode = True
            print("SWITCHED TO ORACLE MODE")
            self._rebuild_oracle_heap()
    
        nv = getattr(req, "next_access_vtime", -1)
        try:
            nv = int(nv)
        except Exception:
            nv = -1

        if nv >= 0:
            self.oracle_nonneg_seen += 1
            if (
                not self.oracle_mode
                and self.oracle_nonneg_seen >= self.cfg.oracle_enable_threshold
            ):
                self.oracle_mode = True
                print("SWITCHED TO ORACLE MODE")
                self._rebuild_oracle_heap()
            return nv

        return INF_NEXT

    def _rebuild_oracle_heap(self) -> None:
        self.max_next_heap.clear()
        for obj_id, rec in self.small.items():
            heappush(self.max_next_heap, (-rec[2], rec[3], obj_id))
        for obj_id, rec in self.main.items():
            heappush(self.max_next_heap, (-rec[2], rec[3], obj_id))

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

    def _add_small(self, obj_id: int, obj_size: int, refbit: int, next_v: int) -> None:
        old = self.small.pop(obj_id, None)
        if old is not None:
            self.small_bytes -= old[0]
        old = self.main.pop(obj_id, None)
        if old is not None:
            self.main_bytes -= old[0]

        rec = [obj_size, refbit, next_v, 0]
        self.small[obj_id] = rec
        self.small_bytes += obj_size
        self._touch_oracle(obj_id, rec, next_v)

    def _add_main(self, obj_id: int, obj_size: int, refbit: int, next_v: int) -> None:
        old = self.main.pop(obj_id, None)
        if old is not None:
            self.main_bytes -= old[0]
        old = self.small.pop(obj_id, None)
        if old is not None:
            self.small_bytes -= old[0]

        rec = [obj_size, refbit, next_v, 0]
        self.main[obj_id] = rec
        self.main_bytes += obj_size
        self._touch_oracle(obj_id, rec, next_v)

    def _touch_oracle(self, obj_id: int, rec, next_v: int) -> None:
        self.stamp += 1
        rec[2] = next_v
        rec[3] = self.stamp
        if self.oracle_mode:
            heappush(self.max_next_heap, (-next_v, self.stamp, obj_id))

    def on_hit(self, req: Request) -> None:
        self.req_seen += 1
        obj_id = int(req.obj_id)
        next_v = self._next_from_req(req)

        rec = self.small.get(obj_id)
        if rec is not None:
            rec[1] = 1
            self._touch_oracle(obj_id, rec, next_v)
            return

        rec = self.main.get(obj_id)
        if rec is not None:
            rec[1] = 1
            self._touch_oracle(obj_id, rec, next_v)
            return

        # Defensive recovery for rare metadata mismatch.
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self._add_main(obj_id, obj_size, 1, next_v)

    def on_miss(self, req: Request) -> None:
        self.req_seen += 1
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)
        next_v = self._next_from_req(req)

        if obj_size > self.cache_size:
            return

        self._remove_obj(obj_id)

        if obj_id in self.ghost:
            self.ghost.pop(obj_id, None)
            self._add_main(obj_id, obj_size, 1, next_v)
        else:
            self._add_small(obj_id, obj_size, 0, next_v)

    def _remove_by_id(self, obj_id: int) -> bool:
        rec = self.small.pop(obj_id, None)
        if rec is not None:
            self.small_bytes -= rec[0]
            self._ghost_add(obj_id)
            return True
        rec = self.main.pop(obj_id, None)
        if rec is not None:
            self.main_bytes -= rec[0]
            self._ghost_add(obj_id)
            return True
        return False

    def _evict_oracle(self) -> Optional[int]:
        while self.max_next_heap:
            neg_next, stamp, obj_id = heappop(self.max_next_heap)
            rec = self.small.get(obj_id)
            if rec is None:
                rec = self.main.get(obj_id)
            if rec is None:
                continue
            if rec[3] != stamp:
                continue
            if self._remove_by_id(obj_id):
                return obj_id
        return None

    def _evict_from_small(self) -> Optional[int]:
        if not self.small:
            return None

        n = len(self.small)
        for _ in range(n):
            obj_id, rec = self.small.popitem(last=False)
            obj_size, refbit, next_v, _stamp = rec
            self.small_bytes -= obj_size

            if refbit:
                self._add_main(obj_id, obj_size, 0, next_v)
                continue

            self._ghost_add(obj_id)
            return obj_id

        return None

    def _evict_from_main(self) -> Optional[int]:
        if not self.main:
            return None

        n = len(self.main)
        for _ in range(n):
            obj_id, rec = self.main.popitem(last=False)
            obj_size, refbit, next_v, _stamp = rec
            self.main_bytes -= obj_size

            if refbit:
                self._add_main(obj_id, obj_size, 0, next_v)
                continue

            self._ghost_add(obj_id)
            return obj_id

        obj_id, rec = self.main.popitem(last=False)
        self.main_bytes -= rec[0]
        self._ghost_add(obj_id)
        return obj_id

    def pick_victim(self, req: Request) -> int:
        _ = req

        if self.oracle_mode:
            victim = self._evict_oracle()
            if victim is not None:
                return victim

        # Fallback selective SIEVE behavior.
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
        self._remove_obj(int(obj_id))

    def on_free(self) -> None:
        self.small.clear()
        self.main.clear()
        self.ghost.clear()
        self.max_next_heap.clear()
        self.small_bytes = 0
        self.main_bytes = 0


def init_hook(common_cache_params: CommonCacheParams) -> OracleBeladyHybrid:
    return OracleBeladyHybrid(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: OracleBeladyHybrid, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: OracleBeladyHybrid, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: OracleBeladyHybrid, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: OracleBeladyHybrid, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: OracleBeladyHybrid) -> None:
    data.on_free()
