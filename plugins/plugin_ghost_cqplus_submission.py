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
        self.p_target_bytes = max(1, int(self.cache_size * 0.10))

        self.p_size_bytes = 0
        self.m_size_bytes = 0

        self.mapping = {}
        self.hand = None

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
            node.visited = True

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        if obj_id in self.mapping:
            return

        if obj_size > self.cache_size:
            return

        # Gentle adaptation using ghost hits.
        delta = max(obj_size, int(self.cache_size * 0.01))

        if obj_id in self.ghost_S:
            self.p_target_bytes = min(int(self.cache_size * 0.40), self.p_target_bytes + delta)
            self.ghost_S_size -= self.ghost_S[obj_id]
            del self.ghost_S[obj_id]
        elif obj_id in self.ghost_M:
            self.p_target_bytes = max(int(self.cache_size * 0.05), self.p_target_bytes - delta)
            self.ghost_M_size -= self.ghost_M[obj_id]
            del self.ghost_M[obj_id]

        # Admit new item into probation.
        node = Node(obj_id, obj_size)
        node.q_type = "probation"
        self._link_head(self.p_head, node)
        self.mapping[obj_id] = node
        self.p_size_bytes += obj_size

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
