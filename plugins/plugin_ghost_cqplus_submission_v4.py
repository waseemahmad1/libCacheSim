from collections import OrderedDict

from libcachesim import CommonCacheParams, Request


class Node:
    __slots__ = ("id", "size", "visited", "q_type", "prev", "next")

    def __init__(self, obj_id: int, size: int):
        self.id = obj_id
        self.size = size
        self.visited = False
        self.q_type = "probation"  # "probation" or "protected"
        self.prev = None
        self.next = None


class GhostCQPlus:
    def __init__(self, cache_size: int):
        self.cache_size = int(cache_size)

        # Probation target starts at 10% of cache bytes.
        self.base_p_target = max(1, int(self.cache_size * 0.10))
        self.min_p_target = max(1, int(self.cache_size * 0.05))
        self.max_p_target = max(self.min_p_target, int(self.cache_size * 0.35))
        self.p_target_bytes = self.base_p_target

        self.p_size_bytes = 0
        self.m_size_bytes = 0

        self.mapping = {}
        self.hand = None

        # Fast windowed adaptation state.
        self.adapt_window = 4096
        self.adapt_step = max(1, int(self.cache_size * 0.01))
        self.req_count = 0
        self.w_misses = 0
        self.w_protected_hits = 0
        self.w_probation_hits = 0
        self.w_ghost_s_hits = 0
        self.w_ghost_m_hits = 0
        self.w_promotions = 0

        # Probation doubly linked list with dummy head/tail.
        self.p_head = Node(-1, 0)
        self.p_tail = Node(-1, 0)
        self.p_head.next = self.p_tail
        self.p_tail.prev = self.p_head

        # Protected doubly linked list with dummy head/tail.
        self.m_head = Node(-1, 0)
        self.m_tail = Node(-1, 0)
        self.m_head.next = self.m_tail
        self.m_tail.prev = self.m_head

        # Ghost histories preserve insertion order.
        self.ghost_S = OrderedDict()
        self.ghost_S_size = 0

        self.ghost_M = OrderedDict()
        self.ghost_M_size = 0

    # ----------------------------
    # Linked-list helpers
    # ----------------------------

    def _link_head(self, head_dummy: Node, node: Node) -> None:
        node.next = head_dummy.next
        node.prev = head_dummy
        head_dummy.next.prev = node
        head_dummy.next = node

    def _unlink(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    # ----------------------------
    # Ghost helpers
    # ----------------------------

    def _add_ghost_S(self, obj_id: int, size: int) -> None:
        if obj_id in self.ghost_S:
            self.ghost_S_size -= self.ghost_S[obj_id]
            del self.ghost_S[obj_id]

        self.ghost_S[obj_id] = size
        self.ghost_S_size += size

        # Cap ghost-S at 100% of cache size.
        while self.ghost_S_size > self.cache_size and self.ghost_S:
            _oldest_id, oldest_size = self.ghost_S.popitem(last=False)
            self.ghost_S_size -= oldest_size

    def _add_ghost_M(self, obj_id: int, size: int) -> None:
        if obj_id in self.ghost_M:
            self.ghost_M_size -= self.ghost_M[obj_id]
            del self.ghost_M[obj_id]

        self.ghost_M[obj_id] = size
        self.ghost_M_size += size

        # Cap ghost-M at 100% of cache size.
        while self.ghost_M_size > self.cache_size and self.ghost_M:
            _oldest_id, oldest_size = self.ghost_M.popitem(last=False)
            self.ghost_M_size -= oldest_size

    def _maybe_adapt(self) -> None:
        if self.req_count < self.adapt_window:
            return

        margin = self.adapt_window >> 4
        pressure_up = self.w_ghost_s_hits + self.w_misses
        pressure_down = (self.w_protected_hits << 1) + self.w_ghost_m_hits + self.w_promotions

        if pressure_up > pressure_down + margin:
            self.p_target_bytes = min(self.max_p_target, self.p_target_bytes + self.adapt_step)
        elif pressure_down > pressure_up + margin:
            self.p_target_bytes = max(self.min_p_target, self.p_target_bytes - self.adapt_step)
        else:
            # Light decay toward a neutral split to react to phase changes.
            if self.p_target_bytes > self.base_p_target:
                self.p_target_bytes = max(self.base_p_target, self.p_target_bytes - self.adapt_step)
            elif self.p_target_bytes < self.base_p_target:
                self.p_target_bytes = min(self.base_p_target, self.p_target_bytes + self.adapt_step)

        self.req_count = 0
        self.w_misses = 0
        self.w_protected_hits = 0
        self.w_probation_hits = 0
        self.w_ghost_s_hits = 0
        self.w_ghost_m_hits = 0
        self.w_promotions = 0

    # ----------------------------
    # Internal eviction apply
    # ----------------------------

    def _evict_probation_node(self, node: Node) -> int:
        self._unlink(node)
        self.p_size_bytes -= node.size
        self._add_ghost_S(node.id, node.size)
        del self.mapping[node.id]
        return node.id

    def _evict_protected_node(self, node: Node) -> int:
        if self.hand == node:
            self.hand = node.prev
            if self.hand == self.m_head:
                self.hand = None

        self._unlink(node)
        self.m_size_bytes -= node.size
        self._add_ghost_M(node.id, node.size)
        del self.mapping[node.id]
        return node.id

    # ----------------------------
    # Policy hooks
    # ----------------------------

    def on_hit(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        node = self.mapping.get(obj_id)
        if node is not None:
            if node.q_type == "protected":
                self.w_protected_hits += 1
                # A repeated protected hit gets refreshed to MRU in protected.
                if node.visited and node is not self.m_head.next:
                    prev_node = node.prev
                    if self.hand == node:
                        self.hand = prev_node
                        if self.hand == self.m_head:
                            self.hand = None
                    self._unlink(node)
                    self._link_head(self.m_head, node)
            else:
                self.w_probation_hits += 1
            node.visited = True

        self.req_count += 1
        self._maybe_adapt()

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        self.req_count += 1
        self.w_misses += 1

        if obj_id in self.mapping:
            self._maybe_adapt()
            return

        if obj_size > self.cache_size:
            self._maybe_adapt()
            return

        # Asymmetric adaptation avoids probation overgrowth in larger caches.
        delta_up = max(obj_size, int(self.cache_size * 0.002))
        delta_down = max(obj_size, int(self.cache_size * 0.004))
        from_ghost_s = False
        from_ghost_m = False

        if obj_id in self.ghost_S:
            self.p_target_bytes = min(self.max_p_target, self.p_target_bytes + delta_up)
            self.ghost_S_size -= self.ghost_S[obj_id]
            del self.ghost_S[obj_id]
            from_ghost_s = True
            self.w_ghost_s_hits += 1
        elif obj_id in self.ghost_M:
            self.p_target_bytes = max(self.min_p_target, self.p_target_bytes - delta_down)
            self.ghost_M_size -= self.ghost_M[obj_id]
            del self.ghost_M[obj_id]
            from_ghost_m = True
            self.w_ghost_m_hits += 1

        # Admission policy:
        # - ghost-M hits: direct protected (strong prior signal)
        # - ghost-S hits: warm probation (avoid over-promotion)
        # - brand-new: cold probation
        node = Node(obj_id, obj_size)
        if from_ghost_m:
            node.q_type = "protected"
            node.visited = True
            self._link_head(self.m_head, node)
            self.m_size_bytes += obj_size
        else:
            node.q_type = "probation"
            node.visited = from_ghost_s
            self._link_head(self.p_head, node)
            self.p_size_bytes += obj_size
        self.mapping[obj_id] = node

        self._maybe_adapt()

    def pick_victim(self, req: Request) -> int:
        _ = req

        while True:
            # Prefer operating on probation when it is at/above target.
            if self.p_size_bytes >= self.p_target_bytes and self.p_tail.prev != self.p_head:
                node = self.p_tail.prev

                if node.visited:
                    node.visited = False
                    self._unlink(node)
                    self.p_size_bytes -= node.size

                    # Promote to protected.
                    node.q_type = "protected"
                    self._link_head(self.m_head, node)
                    self.m_size_bytes += node.size
                    self.w_promotions += 1
                    continue

                # Evict from probation.
                return self._evict_probation_node(node)

            # Otherwise operate on protected.
            if self.m_tail.prev == self.m_head:
                # If protected is empty, force eviction from probation.
                if self.p_tail.prev == self.p_head:
                    if self.mapping:
                        any_id = next(iter(self.mapping))
                        node = self.mapping[any_id]
                        if node.q_type == "probation":
                            return self._evict_probation_node(node)
                        return self._evict_protected_node(node)
                    raise RuntimeError("eviction_hook called with empty cache state")

                node = self.p_tail.prev
                if node.visited:
                    node.visited = False
                    self._unlink(node)
                    self.p_size_bytes -= node.size

                    node.q_type = "protected"
                    self._link_head(self.m_head, node)
                    self.m_size_bytes += node.size
                    self.w_promotions += 1
                    continue

                return self._evict_probation_node(node)

            # Protected segment: SIEVE / CLOCK-like hand scan.
            if self.hand is None or self.hand == self.m_head or self.hand == self.m_tail:
                self.hand = self.m_tail.prev

            while self.hand.visited:
                self.hand.visited = False
                self.hand = self.hand.prev
                if self.hand == self.m_head:
                    self.hand = self.m_tail.prev

            evict_node = self.hand
            # Advance hand before unlinking the victim.
            self.hand = self.hand.prev
            if self.hand == self.m_head:
                self.hand = None

            return self._evict_protected_node(evict_node)

    def on_remove(self, obj_id: int) -> None:
        obj_id = int(obj_id)
        node = self.mapping.get(obj_id)
        if node is None:
            return

        if node.q_type == "probation":
            self._unlink(node)
            self.p_size_bytes -= node.size
        else:
            if self.hand == node:
                self.hand = node.prev
                if self.hand == self.m_head:
                    self.hand = None
            self._unlink(node)
            self.m_size_bytes -= node.size

        del self.mapping[obj_id]

    def on_free(self) -> None:
        self.mapping.clear()
        self.ghost_S.clear()
        self.ghost_M.clear()
        self.hand = None
        self.p_size_bytes = 0
        self.m_size_bytes = 0
        self.ghost_S_size = 0
        self.ghost_M_size = 0
        self.req_count = 0
        self.w_misses = 0
        self.w_protected_hits = 0
        self.w_probation_hits = 0
        self.w_ghost_s_hits = 0
        self.w_ghost_m_hits = 0
        self.w_promotions = 0


def init_hook(params: CommonCacheParams):
    return GhostCQPlus(cache_size=int(params.cache_size))


def hit_hook(data: GhostCQPlus, req: Request):
    data.on_hit(req)


def miss_hook(data: GhostCQPlus, req: Request):
    data.on_miss(req)


def eviction_hook(data: GhostCQPlus, req: Request):
    return data.pick_victim(req)


def remove_hook(data: GhostCQPlus, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: GhostCQPlus):
    data.on_free()
