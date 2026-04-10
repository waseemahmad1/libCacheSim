from collections import OrderedDict

from libcachesim import CommonCacheParams, Request


class Node:
    __slots__ = ("id", "size", "visited", "q_type", "prev", "next")

    def __init__(self, obj_id: int, size: int):
        self.id = obj_id
        self.size = size
        self.visited = False
        self.q_type = "S"  # S (Small), M (Main)
        self.prev = None
        self.next = None


class AdaptiveS3Sieve:
    def __init__(self, cache_size: int):
        self.cache_size = int(cache_size)

        # Adaptation starts at 10%, bounded between 1% and 45%.
        self.s_target_bytes = max(1, int(self.cache_size * 0.10))
        self.s_limit_min = max(1, int(self.cache_size * 0.01))
        self.s_limit_max = max(self.s_limit_min, int(self.cache_size * 0.45))
        self.step_size = max(1, int(self.cache_size * 0.005))

        self.s_size_bytes = 0
        self.m_size_bytes = 0
        self.mapping = {}

        # Small queue (FIFO scan filter).
        self.s_head = Node(-1, 0)
        self.s_tail = Node(-1, 0)
        self.s_head.next = self.s_tail
        self.s_tail.prev = self.s_head

        # Main queue (SIEVE-like).
        self.m_head = Node(-1, 0)
        self.m_tail = Node(-1, 0)
        self.m_head.next = self.m_tail
        self.m_tail.prev = self.m_head
        self.sieve_hand = None

        # Ghost queue tracks IDs evicted from S.
        self.ghost = OrderedDict()
        self.ghost_size = 0

    def _link_head(self, head_dummy: Node, node: Node) -> None:
        node.next = head_dummy.next
        node.prev = head_dummy
        head_dummy.next.prev = node
        head_dummy.next = node

    def _unlink(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        if self.sieve_hand == node:
            self.sieve_hand = node.prev if node.prev.id != -1 else None
        node.prev = None
        node.next = None

    def _add_ghost(self, obj_id: int, size: int) -> None:
        old_size = self.ghost.pop(obj_id, None)
        if old_size is not None:
            self.ghost_size -= old_size

        self.ghost[obj_id] = size
        self.ghost_size += size

        while self.ghost_size > self.cache_size and self.ghost:
            _, g_sz = self.ghost.popitem(last=False)
            self.ghost_size -= g_sz

    def on_hit(self, req: Request) -> None:
        node = self.mapping.get(int(req.obj_id))
        if node is not None:
            node.visited = True

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        if obj_id in self.mapping:
            return
        if obj_size > self.cache_size:
            return

        if obj_id in self.ghost:
            # Ghost hit means S was too small: grow S target and admit to M.
            self.s_target_bytes = min(self.s_limit_max, self.s_target_bytes + self.step_size)
            self.ghost_size -= self.ghost.pop(obj_id)

            node = Node(obj_id, obj_size)
            node.q_type = "M"
            self._link_head(self.m_head, node)
            self.m_size_bytes += obj_size
        else:
            node = Node(obj_id, obj_size)
            node.q_type = "S"
            self._link_head(self.s_head, node)
            self.s_size_bytes += obj_size

        self.mapping[obj_id] = node

    def pick_victim(self, req: Request) -> int:
        _ = req
        while self.s_size_bytes + self.m_size_bytes >= self.cache_size:
            # Rule 1: If S is over target, evict/promo from S.
            if self.s_size_bytes >= self.s_target_bytes and self.s_tail.prev != self.s_head:
                victim = self.s_tail.prev
                self._unlink(victim)
                self.s_size_bytes -= victim.size

                if victim.visited:
                    victim.visited = False
                    victim.q_type = "M"
                    self._link_head(self.m_head, victim)
                    self.m_size_bytes += victim.size
                    continue

                self._add_ghost(victim.id, victim.size)
                del self.mapping[victim.id]
                return victim.id

            # Rule 2: Evict from M with SIEVE hand.
            if self.m_tail.prev != self.m_head:
                if self.sieve_hand is None or self.sieve_hand == self.m_head:
                    self.sieve_hand = self.m_tail.prev

                while self.sieve_hand is not None and self.sieve_hand != self.m_head:
                    if self.sieve_hand.visited:
                        self.sieve_hand.visited = False
                        self.sieve_hand = self.sieve_hand.prev
                        if self.sieve_hand == self.m_head:
                            self.sieve_hand = self.m_tail.prev
                    else:
                        victim = self.sieve_hand
                        self.sieve_hand = victim.prev
                        if self.sieve_hand == self.m_head:
                            self.sieve_hand = None
                        self._unlink(victim)
                        self.m_size_bytes -= victim.size
                        del self.mapping[victim.id]
                        return victim.id

            # Rule 3: If M empty, force S fallback.
            if self.s_tail.prev != self.s_head:
                victim = self.s_tail.prev
                self._unlink(victim)
                self.s_size_bytes -= victim.size
                self._add_ghost(victim.id, victim.size)
                del self.mapping[victim.id]
                return victim.id

            if self.mapping:
                any_id, victim = next(iter(self.mapping.items()))
                self._unlink(victim)
                if victim.q_type == "S":
                    self.s_size_bytes -= victim.size
                else:
                    self.m_size_bytes -= victim.size
                del self.mapping[any_id]
                return any_id
            raise RuntimeError("eviction_hook called with empty cache state")

        if self.mapping:
            # Defensive fallback for validators that may call eviction out of order.
            any_id = next(iter(self.mapping))
            return any_id
        raise RuntimeError("eviction_hook called with empty cache state")

    def on_remove(self, obj_id: int) -> None:
        node = self.mapping.get(int(obj_id))
        if node is None:
            return

        self._unlink(node)
        if node.q_type == "S":
            self.s_size_bytes -= node.size
        else:
            self.m_size_bytes -= node.size
        del self.mapping[node.id]

    def on_free(self) -> None:
        self.mapping.clear()
        self.ghost.clear()
        self.sieve_hand = None
        self.s_size_bytes = 0
        self.m_size_bytes = 0
        self.ghost_size = 0


def init_hook(params: CommonCacheParams):
    return AdaptiveS3Sieve(int(params.cache_size))


def hit_hook(data: AdaptiveS3Sieve, req: Request):
    data.on_hit(req)


def miss_hook(data: AdaptiveS3Sieve, req: Request):
    data.on_miss(req)


def eviction_hook(data: AdaptiveS3Sieve, req: Request):
    return data.pick_victim(req)


def remove_hook(data: AdaptiveS3Sieve, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: AdaptiveS3Sieve):
    data.on_free()
