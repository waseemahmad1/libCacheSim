from collections import OrderedDict
from math import log, sqrt
from typing import Optional

from libcachesim import CommonCacheParams, Request


class PolicyConfig:
    def __init__(
        self,
        probation_frac_arms=(0.12, 0.18, 0.24, 0.32, 0.42, 0.52),
        adapt_every: int = 2048,
        short_gap_epochs: int = 2,
        ghost_factor: int = 6,
        min_ghost_entries: int = 4096,
        max_ghost_entries: int = 500000,
        ucb_c: float = 0.18,
        reward_ema_alpha: float = 0.25,
    ):
        self.probation_frac_arms = probation_frac_arms
        self.adapt_every = adapt_every
        self.short_gap_epochs = short_gap_epochs
        self.ghost_factor = ghost_factor
        self.min_ghost_entries = min_ghost_entries
        self.max_ghost_entries = max_ghost_entries
        self.ucb_c = ucb_c
        self.reward_ema_alpha = reward_ema_alpha


class SelfTuningS3Sieve:

    def __init__(self, cache_size: int, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.cache_size = int(cache_size)

        # obj_id -> [size, refbit, last_epoch, ghost_mark]
        self.probation = OrderedDict()
        # obj_id -> [size, refbit]
        self.protected = OrderedDict()
        # obj_id -> origin ('P' or 'R')
        self.ghost = OrderedDict()

        self.probation_bytes = 0
        self.protected_bytes = 0

        self.req_count = 0
        self.epoch = 0

        ghost_cap = self.cache_size * self.cfg.ghost_factor
        self.ghost_limit = max(
            self.cfg.min_ghost_entries, min(self.cfg.max_ghost_entries, ghost_cap)
        )

        self.frac_arms = tuple(self.cfg.probation_frac_arms)
        self.n_arms = len(self.frac_arms)
        self.arm_pulls = [0] * self.n_arms
        self.arm_value = [0.0] * self.n_arms
        self.arm_idx = min(2, self.n_arms - 1)  # start near moderate probation
        self.total_adapt_rounds = 0

        self.probation_target = self._target_from_arm(self.arm_idx)
        self.target_step = max(1, self.cache_size // 50)

        # Short-window stats for reward/adaptation.
        self.win_hit_probation = 0
        self.win_hit_protected = 0
        self.win_miss = 0
        self.win_ghost_from_probation = 0
        self.win_ghost_from_protected = 0
        self.scan_mode = False

    def _target_from_arm(self, idx: int) -> int:
        frac = self.frac_arms[idx]
        lo = max(1, int(self.cache_size * min(self.frac_arms)))
        hi = max(lo, int(self.cache_size * max(self.frac_arms)))
        val = int(self.cache_size * frac)
        return min(hi, max(lo, val))

    def _tick(self) -> None:
        self.req_count += 1
        if (self.req_count & 127) == 0:
            self.epoch += 1
        if self.req_count % self.cfg.adapt_every == 0:
            self._adapt_bandit()

    def _adapt_bandit(self) -> None:
        # Compute reward of current arm from short-window outcomes.
        # Emphasize protected hits; penalize misses and probation-ghost pressure.
        reward = (
            2.0 * self.win_hit_protected
            + 0.8 * self.win_hit_probation
            - 1.6 * self.win_miss
            - 1.2 * self.win_ghost_from_probation
            - 0.4 * self.win_ghost_from_protected
        ) / float(max(1, self.cfg.adapt_every))

        i = self.arm_idx
        self.arm_pulls[i] += 1
        a = self.cfg.reward_ema_alpha
        self.arm_value[i] = (1.0 - a) * self.arm_value[i] + a * reward

        self.total_adapt_rounds += 1

        # Fast scan signal for admission placement.
        hits = self.win_hit_probation + self.win_hit_protected
        self.scan_mode = self.win_miss > (hits * 3 + 64)

        # Explore each arm at least once, then UCB selection.
        if self.total_adapt_rounds < self.n_arms:
            self.arm_idx = self.total_adapt_rounds
        else:
            log_term = log(float(self.total_adapt_rounds) + 1.0)
            best_idx = 0
            best_score = -1e18
            for j in range(self.n_arms):
                pulls = self.arm_pulls[j]
                # Small positive denominator avoids division-by-zero.
                bonus = self.cfg.ucb_c * sqrt(log_term / (pulls + 1e-9))
                score = self.arm_value[j] + bonus
                if score > best_score:
                    best_score = score
                    best_idx = j
            self.arm_idx = best_idx

        # Apply selected target; small scan-time nudge toward larger probation.
        self.probation_target = self._target_from_arm(self.arm_idx)
        if self.scan_mode:
            self.probation_target = min(
                self.cache_size - 1, self.probation_target + self.target_step
            )

        # Reset window counters.
        self.win_hit_probation = 0
        self.win_hit_protected = 0
        self.win_miss = 0
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

    def _add_probation(
        self, obj_id: int, obj_size: int, refbit: int, ghost_mark: int, near_head: bool
    ) -> None:
        self._remove_obj(obj_id)
        self.probation[obj_id] = [obj_size, refbit, self.epoch, ghost_mark]
        if near_head:
            # In scan phases, put new items closer to eviction edge.
            self.probation.move_to_end(obj_id, last=False)
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
            self.win_hit_probation += 1
            obj_size, refbit, last_epoch, ghost_mark = rec
            short_gap = (self.epoch - last_epoch) <= self.cfg.short_gap_epochs
            # Promote only on clear quick reuse evidence.
            if short_gap and (ghost_mark == 1 or refbit == 1):
                self.probation.pop(obj_id, None)
                self.probation_bytes -= obj_size
                self._add_protected(obj_id, obj_size, 1)
            else:
                rec[1] = 1
                rec[2] = self.epoch
                rec[3] = 0
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
            # Ghost hits re-enter probation warm; no direct protected admission.
            self._add_probation(obj_id, obj_size, 1, 1, near_head=False)
            return

        near_head = self.scan_mode and self.probation_bytes >= self.probation_target
        self._add_probation(obj_id, obj_size, 0, 0, near_head=near_head)

    def _evict_from_probation(self) -> Optional[int]:
        if not self.probation:
            return None

        n = len(self.probation)
        for _ in range(n):
            obj_id, rec = self.probation.popitem(last=False)
            obj_size, refbit, _last_epoch, _ghost_mark = rec
            self.probation_bytes -= obj_size

            if refbit == 0:
                self._ghost_add(obj_id, "P")
                return obj_id

            # One simple second chance in-place, then cold again.
            self.probation[obj_id] = [obj_size, 0, self.epoch, 0]
            self.probation_bytes += obj_size

        # Guarantee progress.
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

            self.protected[obj_id] = [obj_size, 0]
            self.protected_bytes += obj_size

        # Guarantee progress.
        obj_id, rec = self.protected.popitem(last=False)
        self.protected_bytes -= rec[0]
        self._ghost_add(obj_id, "R")
        return obj_id

    def pick_victim(self, req: Request) -> int:
        _ = req

        protected_target = max(1, self.cache_size - self.probation_target)

        # If protected exceeds target by a lot, pressure protected first.
        if self.protected and self.protected_bytes > (protected_target + self.target_step):
            victim = self._evict_from_protected()
            if victim is not None:
                return victim

        # Default: clear probation first for scan resistance.
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


def init_hook(common_cache_params: CommonCacheParams) -> SelfTuningS3Sieve:
    return SelfTuningS3Sieve(cache_size=int(common_cache_params.cache_size))


def hit_hook(data: SelfTuningS3Sieve, req: Request) -> None:
    data.on_hit(req)


def miss_hook(data: SelfTuningS3Sieve, req: Request) -> None:
    data.on_miss(req)


def eviction_hook(data: SelfTuningS3Sieve, req: Request) -> int:
    return data.pick_victim(req)


def remove_hook(data: SelfTuningS3Sieve, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: SelfTuningS3Sieve) -> None:
    data.on_free()
