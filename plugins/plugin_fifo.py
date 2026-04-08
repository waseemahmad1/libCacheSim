"""
Minimal FIFO baseline plugin for libCacheSim Python hooks.

This file is intentionally simple and uses the same CLI arguments as
plugins/plugin_competition.py for direct miss-ratio comparison.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from libcachesim import CommonCacheParams, PluginCache, Request, TraceReader, TraceType


class FifoPolicy:
    def __init__(self, cache_size: int):
        self.cache_size = int(cache_size)
        # FIFO order: oldest -> newest
        self.queue: "OrderedDict[int, int]" = OrderedDict()

    def on_hit(self, req: Request) -> None:
        # FIFO does not reorder on hit.
        _ = req

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        # Oversized objects are not inserted by core cache; skip metadata.
        if obj_size > self.cache_size:
            return

        # Defensive dedupe in case metadata got out of sync.
        self.queue.pop(obj_id, None)
        self.queue[obj_id] = obj_size

    def pick_victim(self, req: Request) -> int:
        _ = req
        if not self.queue:
            raise RuntimeError("FIFO metadata empty in eviction_hook")
        victim_id, _ = self.queue.popitem(last=False)
        return victim_id

    def on_remove(self, obj_id: int) -> None:
        # Tolerant of missing IDs.
        self.queue.pop(int(obj_id), None)

    def on_free(self) -> None:
        self.queue.clear()


# ---------------------------
# libCacheSim hook functions
# ---------------------------

def cache_init_hook(common_cache_params: CommonCacheParams) -> FifoPolicy:
    return FifoPolicy(cache_size=int(common_cache_params.cache_size))


def cache_hit_hook(data: FifoPolicy, req: Request) -> None:
    data.on_hit(req)


def cache_miss_hook(data: FifoPolicy, req: Request) -> None:
    data.on_miss(req)


def cache_eviction_hook(data: FifoPolicy, req: Request) -> int:
    return data.pick_victim(req)


def cache_remove_hook(data: FifoPolicy, obj_id: int) -> None:
    data.on_remove(obj_id)


def cache_free_hook(data: FifoPolicy) -> None:
    data.on_free()


# ---------------------------
# Local test entry point
# ---------------------------

def _parse_size(size_text: str) -> int:
    s = size_text.strip().lower()
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
        cache_name="fifo_baseline",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run local FIFO baseline plugin")
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

    print("Policy: fifo_baseline")
    print(f"Trace: {args.trace}")
    print(f"Cache size (bytes): {cache_size_bytes}")
    print(f"Request miss ratio: {req_miss_ratio:.6f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.6f}")
