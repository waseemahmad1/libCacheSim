from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class Node:
    __slots__ = ("id", "size", "visited", "q_type", "prev", "next")

    def __init__(self, obj_id: int, size: int):
        self.id = obj_id
        self.size = size
        self.visited = False
        self.q_type = "probation"
        self.prev = None
        self.next = None


class AdaptiveSegmentedLRU:
    def __init__(self, cache_size: int):
        self.cache_size = int(cache_size)

        # p_target is the target size for the Probation segment.
        # It adapts based on ghost hits.
        self.p_target = max(1, int(self.cache_size * 0.10))

        self.p_size = 0
        self.m_size = 0
        self.mapping = {}

        # Probation Queue (FIFO with Lazy Promotion)
        self.p_head = Node(-1, 0)
        self.p_tail = Node(-1, 0)
        self.p_head.next = self.p_tail
        self.p_tail.prev = self.p_head

        # Protected Queue (LRU-like main working set)
        self.m_head = Node(-1, 0)
        self.m_tail = Node(-1, 0)
        self.m_head.next = self.m_tail
        self.m_tail.prev = self.m_head

        # Ghost queues to track evicted metadata
        self.ghost_S = OrderedDict()
        self.ghost_S_size = 0
        self.ghost_M = OrderedDict()
        self.ghost_M_size = 0

    def _link_head(self, head_dummy: Node, node: Node) -> None:
        node.next = head_dummy.next
        node.prev = head_dummy
        head_dummy.next.prev = node
        head_dummy.next = node

    def _unlink(self, node: Node) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_ghost(self, node: Node, q_type: str) -> None:
        if q_type == "S":
            if node.id in self.ghost_S:
                self.ghost_S_size -= self.ghost_S[node.id]
                del self.ghost_S[node.id]
            self.ghost_S[node.id] = node.size
            self.ghost_S_size += node.size
            while self.ghost_S_size > self.cache_size and self.ghost_S:
                _oid, sz = self.ghost_S.popitem(last=False)
                self.ghost_S_size -= sz
        else:
            if node.id in self.ghost_M:
                self.ghost_M_size -= self.ghost_M[node.id]
                del self.ghost_M[node.id]
            self.ghost_M[node.id] = node.size
            self.ghost_M_size += node.size
            while self.ghost_M_size > self.cache_size and self.ghost_M:
                _oid, sz = self.ghost_M.popitem(last=False)
                self.ghost_M_size -= sz

    def _evict_node(self, node: Node, q_type: str) -> int:
        self._unlink(node)
        if q_type == "S":
            self.p_size -= node.size
        else:
            self.m_size -= node.size
        self._add_ghost(node, q_type)
        del self.mapping[node.id]
        return node.id

    def _admit(self, obj_id: int, size: int, q_type: str, visited: bool) -> None:
        # Defensive dedupe in case evaluator hooks call unexpectedly.
        old = self.mapping.pop(obj_id, None)
        if old is not None:
            self._unlink(old)
            if old.q_type == "probation":
                self.p_size -= old.size
            else:
                self.m_size -= old.size

        node = Node(obj_id, size)
        node.q_type = q_type
        node.visited = visited
        self.mapping[obj_id] = node

        if q_type == "probation":
            self._link_head(self.p_head, node)
            self.p_size += size
        else:
            self._link_head(self.m_head, node)
            self.m_size += size

    def on_hit(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        node = self.mapping.get(obj_id)
        if node is None:
            return

        node.visited = True

        if node.q_type == "protected":
            self._unlink(node)
            self._link_head(self.m_head, node)

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        if obj_id in self.mapping:
            return

        if obj_size > self.cache_size:
            return

        delta = max(obj_size, int(self.cache_size * 0.002))

        if obj_id in self.ghost_S:
            self.p_target = min(int(self.cache_size * 0.95), self.p_target + delta)
            self.ghost_S_size -= self.ghost_S.pop(obj_id)
            # Ghost re-reference: bypass probation.
            self._admit(obj_id, obj_size, "protected", visited=True)
            return

        if obj_id in self.ghost_M:
            # Item was recently in Protected. Shrink Probation target.
            self.p_target = max(int(self.cache_size * 0.05), self.p_target - delta)
            self.ghost_M_size -= self.ghost_M.pop(obj_id)
            self._admit(obj_id, obj_size, "protected", visited=True)
            return

        # new item.
        self._admit(obj_id, obj_size, "probation", visited=False)

    def pick_victim(self, req: Request) -> int:
        _ = req

        while True:
            if (self.p_size >= self.p_target or self.m_size == 0) and self.p_tail.prev != self.p_head:
                node = self.p_tail.prev
                if node.visited:
                    # Lazy Promotion: survived probation, move to Protected.
                    node.visited = False
                    self._unlink(node)
                    self.p_size -= node.size
                    node.q_type = "protected"
                    self._link_head(self.m_head, node)
                    self.m_size += node.size
                    continue

                return self._evict_node(node, "S")

            if self.m_tail.prev != self.m_head:
                node = self.m_tail.prev
                if node.visited:
                    node.visited = False
                    self._unlink(node)
                    self._link_head(self.m_head, node)
                    continue

                return self._evict_node(node, "M")

            if self.p_tail.prev != self.p_head:
                return self._evict_node(self.p_tail.prev, "S")

            if self.mapping:
                any_id = next(iter(self.mapping))
                node = self.mapping[any_id]
                if node.q_type == "probation":
                    return self._evict_node(node, "S")
                return self._evict_node(node, "M")

            raise RuntimeError("eviction_hook called with empty cache state")

    def on_remove(self, obj_id: int) -> None:
        obj_id = int(obj_id)
        node = self.mapping.pop(obj_id, None)
        if node is None:
            return

        self._unlink(node)
        if node.q_type == "probation":
            self.p_size -= node.size
        else:
            self.m_size -= node.size

    def on_free(self) -> None:
        self.mapping.clear()
        self.ghost_S.clear()
        self.ghost_M.clear()
        self.p_size = 0
        self.m_size = 0
        self.ghost_S_size = 0
        self.ghost_M_size = 0


def init_hook(params: CommonCacheParams):
    return AdaptiveSegmentedLRU(cache_size=int(params.cache_size))


def hit_hook(data: AdaptiveSegmentedLRU, req: Request):
    data.on_hit(req)


def miss_hook(data: AdaptiveSegmentedLRU, req: Request):
    data.on_miss(req)


def eviction_hook(data: AdaptiveSegmentedLRU, req: Request):
    return data.pick_victim(req)


def remove_hook(data: AdaptiveSegmentedLRU, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: AdaptiveSegmentedLRU):
    data.on_free()
