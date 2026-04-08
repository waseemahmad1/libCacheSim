from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from libcachesim import CommonCacheParams, PluginCache, Request, TraceReader, TraceType

@dataclass
class PolicyConfig:
    window_fraction: float = 0.20
    min_window_bytes: int = 64 * 1024
    ghost_factor: int = 4
    min_ghost_entries: int = 10_000
    max_ghost_entries: int = 400_000


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

def cache_init_hook(common_cache_params: CommonCacheParams) -> Competition2QGhost:
    return Competition2QGhost(cache_size=int(common_cache_params.cache_size))


def cache_hit_hook(data: Competition2QGhost, req: Request) -> None:
    data.on_hit(req)


def cache_miss_hook(data: Competition2QGhost, req: Request) -> None:
    data.on_miss(req)


def cache_eviction_hook(data: Competition2QGhost, req: Request) -> int:
    return data.pick_victim(req)


def cache_remove_hook(data: Competition2QGhost, obj_id: int) -> None:
    data.on_remove(obj_id)


def cache_free_hook(data: Competition2QGhost) -> None:
    data.on_free()

def _parse_size(size_text: str) -> int:
    s = size_text.strip().lower()
    # Match longer suffixes first so "64mb" does not get parsed as "64m" + "b".
    units = [
        ("gb", 1024**3),
        ("g", 1024**3),
        ("mb", 1024**2),
        ("m", 1024**2),
        ("kb", 1024),
        ("k", 1024),
        ("b", 1),
    ]
    for u, mult in units:
        if s.endswith(u):
            num = float(s[: -len(u)] or "0")
            return int(num * mult)
    return int(float(s))


def _trace_type_from_text(name: str) -> Any:
    name = name.strip().lower()
    # Support common aliases used in docs.
    aliases = {
        "vscsi": ["VSCSI_TRACE", "VSCSI"],
        "csv": ["CSV_TRACE", "CSV"],
        "txt": ["TXT_TRACE", "TXT"],
        "oraclegeneral": ["ORACLE_GENERAL_TRACE", "ORACLE_GENERAL"],
    }
    keys = aliases.get(name, [name.upper()])
    for key in keys:
        if hasattr(TraceType, key):
            return getattr(TraceType, key)
    raise ValueError(f"Unsupported trace type: {name}")


def _build_plugin(cache_size_bytes: int) -> PluginCache:
    return PluginCache(
        cache_size=cache_size_bytes,
        cache_init_hook=cache_init_hook,
        cache_hit_hook=cache_hit_hook,
        cache_miss_hook=cache_miss_hook,
        cache_eviction_hook=cache_eviction_hook,
        cache_remove_hook=cache_remove_hook,
        cache_free_hook=cache_free_hook,
        cache_name="competition_2qghost",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run local test for competition plugin")
    parser.add_argument(
        "--trace",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "cloudPhysicsIO.vscsi"),
        help="Path to trace file",
    )
    parser.add_argument(
        "--trace-type",
        type=str,
        default="vscsi",
        help="Trace type (vscsi/csv/txt/oracleGeneral)",
    )
    parser.add_argument(
        "--cache-size",
        type=str,
        default="64mb",
        help="Cache size in bytes or with unit (e.g., 64mb)",
    )
    args = parser.parse_args()

    cache_size_bytes = _parse_size(args.cache_size)
    trace_type = _trace_type_from_text(args.trace_type)

    plugin_cache = _build_plugin(cache_size_bytes)
    reader = TraceReader(trace=args.trace, trace_type=trace_type)
    req_miss_ratio, byte_miss_ratio = plugin_cache.process_trace(reader)

    print("Policy: competition_2qghost")
    print(f"Trace: {args.trace}")
    print(f"Cache size (bytes): {cache_size_bytes}")
    print(f"Request miss ratio: {req_miss_ratio:.6f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.6f}")
