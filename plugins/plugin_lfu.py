from collections import OrderedDict
from typing import Optional

from libcachesim import CommonCacheParams, Request

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class _CountMinSketch:

    _SEEDS = (0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F)
    _MAX = 15  # 4-bit saturation cap

    def __init__(self, size: int) -> None:
        # Width rounded up to next power-of-two for fast modulo via bit-mask
        self.w = _next_pow2(max(64, size))
        self._mask = self.w - 1
        self._rows = [[0] * self.w for _ in range(4)]
        self._n: int = 0
        self._reset_at: int = self.w   # halve after w increments

    def _idx(self, key: int, seed: int) -> int:
        h = (key * seed) & 0xFFFF_FFFF_FFFF_FFFF
        h ^= h >> 32
        return int(h) & self._mask

    def increment(self, key: int) -> None:
        for i, s in enumerate(self._SEEDS):
            j = self._idx(key, s)
            if self._rows[i][j] < self._MAX:
                self._rows[i][j] += 1
        self._n += 1
        if self._n >= self._reset_at:
            self._age()

    def estimate(self, key: int) -> int:
        return min(self._rows[i][self._idx(key, s)]
                   for i, s in enumerate(self._SEEDS))

    def _age(self) -> None:
        """Halve all counters to discount old accesses."""
        for row in self._rows:
            for j in range(self.w):
                row[j] >>= 1
        self._n >>= 1


# ── W-TinyLFU cache ───────────────────────────────────────────────────────────

class WTinyLFUCache:
    """Byte-aware W-TinyLFU cache policy."""

    _W_FRAC: float = 0.10   # window fraction of total cache bytes
    _P_FRAC: float = 0.80   # protected fraction of main bytes

    def __init__(self, cache_size: int) -> None:
        self.cache_size = cache_size

        self.w_target: int = max(1, int(cache_size * self._W_FRAC))
        main: int = max(1, cache_size - self.w_target)
        self.p_target: int = max(1, int(main * self._P_FRAC))

        # Three LRU segments; OrderedDict order = oldest first (LRU end)
        self.W: OrderedDict[int, int] = OrderedDict()   # obj_id → size
        self.P: OrderedDict[int, int] = OrderedDict()
        self.Q: OrderedDict[int, int] = OrderedDict()

        self.W_bytes: int = 0
        self.P_bytes: int = 0
        self.Q_bytes: int = 0

        self.sketch = _CountMinSketch(max(64, cache_size))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _remove_any(self, obj_id: int) -> None:
        """Remove obj_id from whichever segment holds it."""
        sz = self.W.pop(obj_id, None)
        if sz is not None:
            self.W_bytes -= sz
            return
        sz = self.P.pop(obj_id, None)
        if sz is not None:
            self.P_bytes -= sz
            return
        sz = self.Q.pop(obj_id, None)
        if sz is not None:
            self.Q_bytes -= sz

    def _demote_p_overflow(self) -> None:
        """Move P's LRU to Q's MRU while P exceeds its byte target."""
        while self.P_bytes > self.p_target and self.P:
            oid, sz = self.P.popitem(last=False)
            self.P_bytes -= sz
            self.Q[oid] = sz
            self.Q_bytes += sz

    # ── Public hooks ──────────────────────────────────────────────────────────

    def on_hit(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        self.sketch.increment(obj_id)

        # W hit: refresh recency (move to MRU end)
        if obj_id in self.W:
            sz = self.W.pop(obj_id)
            self.W[obj_id] = sz
            return

        # Q hit: promote to P MRU
        sz = self.Q.pop(obj_id, None)
        if sz is not None:
            self.Q_bytes -= sz
            self.P[obj_id] = sz
            self.P_bytes += sz
            self._demote_p_overflow()
            return

        # P hit: move to P MRU
        sz = self.P.pop(obj_id, None)
        if sz is not None:
            self.P_bytes -= sz
            self.P[obj_id] = sz
            self.P_bytes += sz
            return

        # Rare metadata desync: re-admit as hot
        obj_size = int(req.obj_size)
        if obj_size <= self.cache_size:
            self.P[obj_id] = obj_size
            self.P_bytes += obj_size
            self._demote_p_overflow()

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        # Oversized: core cache won't insert; skip tracking
        if obj_size > self.cache_size:
            return

        self.sketch.increment(obj_id)
        self._remove_any(obj_id)   # stale-metadata guard

        # All new objects enter via W at MRU
        self.W[obj_id] = obj_size
        self.W_bytes += obj_size

    def pick_victim(self, req: Request) -> int:
        # ── Phase 1: Admission filter when W exceeds its byte budget ─────────
        if self.W_bytes > self.w_target and self.W:
            # W's eviction candidate (oldest in W)
            wc_id, wc_sz = next(iter(self.W.items()))

            # Main-segment competitor: prefer Q's LRU; fall back to P's LRU
            if self.Q:
                mc_id, mc_sz = next(iter(self.Q.items()))
                mc_in_q = True
            elif self.P:
                mc_id, mc_sz = next(iter(self.P.items()))
                mc_in_q = False
            else:
                # Main is empty; evict W's LRU unconditionally
                self.W.popitem(last=False)
                self.W_bytes -= wc_sz
                return wc_id

            w_freq = self.sketch.estimate(wc_id)
            m_freq = self.sketch.estimate(mc_id)

            if w_freq >= m_freq:
                # W candidate is more valuable: promote it to Q, evict mc
                self.W.popitem(last=False)
                self.W_bytes -= wc_sz
                self.Q[wc_id] = wc_sz          # insert at Q's MRU end
                self.Q_bytes += wc_sz

                if mc_in_q:
                    del self.Q[mc_id]           # mc is still at Q's LRU end
                    self.Q_bytes -= mc_sz
                else:
                    del self.P[mc_id]
                    self.P_bytes -= mc_sz
                return mc_id
            else:
                # Main competitor is more valuable: reject W candidate
                self.W.popitem(last=False)
                self.W_bytes -= wc_sz
                return wc_id

        # ── Phase 2: W is within budget; evict from main ─────────────────────
        self._demote_p_overflow()

        if self.Q:
            oid, sz = self.Q.popitem(last=False)
            self.Q_bytes -= sz
            return oid

        if self.P:
            oid, sz = self.P.popitem(last=False)
            self.P_bytes -= sz
            return oid

        # Last resort: drain W
        if self.W:
            oid, sz = self.W.popitem(last=False)
            self.W_bytes -= sz
            return oid

        raise RuntimeError("WTinyLFU: eviction_hook called with all queues empty")

    def on_remove(self, obj_id: int) -> None:
        self._remove_any(int(obj_id))

    def on_free(self) -> None:
        self.W.clear()
        self.P.clear()
        self.Q.clear()
        self.W_bytes = 0
        self.P_bytes = 0
        self.Q_bytes = 0


# ── libcachesim hook interface ────────────────────────────────────────────────

def init_hook(common_cache_params: CommonCacheParams) -> WTinyLFUCache:
    return WTinyLFUCache(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: WTinyLFUCache, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: WTinyLFUCache, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: WTinyLFUCache, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: WTinyLFUCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: WTinyLFUCache) -> None:
    data.on_free()