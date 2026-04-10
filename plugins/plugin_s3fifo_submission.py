from collections import OrderedDict

from libcachesim import CommonCacheParams, Request


class Node:
    __slots__ = ("id", "size", "freq", "q_type", "prev", "next")

    def __init__(self, obj_id: int, size: int):
        self.id = obj_id
        self.size = size
        self.freq = 0
        self.q_type = "S"
        self.prev = None
        self.next = None


class S3FIFO:
    def __init__(self, cache_size: int):
        self.cache_size = int(cache_size)

        # Small queue target is fixed at 10% of total cache bytes.
        self.s_target_bytes = max(1, int(self.cache_size * 0.10))

        self.s_size_bytes = 0
        self.m_size_bytes = 0
        self.mapping = {}

        # Small/probation queue.
        self.s_head = Node(-1, 0)
        self.s_tail = Node(-1, 0)
        self.s_head.next = self.s_tail
        self.s_tail.prev = self.s_head

        # Main/protected queue.
        self.m_head = Node(-1, 0)
        self.m_tail = Node(-1, 0)
        self.m_head.next = self.m_tail
        self.m_tail.prev = self.m_head

        # Ghost queue tracks metadata for objects evicted from S.
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
        node.prev = None
        node.next = None

    def _add_ghost(self, obj_id: int, size: int) -> None:
        old_size = self.ghost.pop(obj_id, None)
        if old_size is not None:
            self.ghost_size -= old_size

        self.ghost[obj_id] = size
        self.ghost_size += size

        while self.ghost_size > self.cache_size and self.ghost:
            _, evict_size = self.ghost.popitem(last=False)
            self.ghost_size -= evict_size

    def on_hit(self, req: Request) -> None:
        node = self.mapping.get(int(req.obj_id))
        if node is not None:
            # Cap frequency so stale objects cannot survive forever.
            node.freq = min(node.freq + 1, 3)

    def on_miss(self, req: Request) -> None:
        obj_id = int(req.obj_id)
        obj_size = int(req.obj_size)

        if obj_id in self.mapping:
            return
        if obj_size > self.cache_size:
            return

        node = Node(obj_id, obj_size)

        if obj_id in self.ghost:
            self.ghost_size -= self.ghost.pop(obj_id)
            node.q_type = "M"
            node.freq = 0
            self._link_head(self.m_head, node)
            self.m_size_bytes += obj_size
        else:
            node.q_type = "S"
            node.freq = 0
            self._link_head(self.s_head, node)
            self.s_size_bytes += obj_size

        self.mapping[obj_id] = node

    def pick_victim(self, req: Request) -> int:
        _ = req
        while True:
            # 1) Drain S when over target.
            if self.s_size_bytes >= self.s_target_bytes and self.s_tail.prev != self.s_head:
                node = self.s_tail.prev
                if node.freq > 0:
                    self._unlink(node)
                    self.s_size_bytes -= node.size
                    node.freq = 0
                    node.q_type = "M"
                    self._link_head(self.m_head, node)
                    self.m_size_bytes += node.size
                    continue

                self._unlink(node)
                self.s_size_bytes -= node.size
                self._add_ghost(node.id, node.size)
                del self.mapping[node.id]
                return node.id

            # 2) Evict from M using queue-clock behavior.
            if self.m_tail.prev != self.m_head:
                node = self.m_tail.prev
                if node.freq > 0:
                    self._unlink(node)
                    node.freq -= 1
                    self._link_head(self.m_head, node)
                    continue

                self._unlink(node)
                self.m_size_bytes -= node.size
                del self.mapping[node.id]
                return node.id

            # 3) Fallback when M is empty: drain S.
            if self.s_tail.prev != self.s_head:
                node = self.s_tail.prev
                if node.freq > 0:
                    self._unlink(node)
                    self.s_size_bytes -= node.size
                    node.freq = 0
                    node.q_type = "M"
                    self._link_head(self.m_head, node)
                    self.m_size_bytes += node.size
                    continue

                self._unlink(node)
                self.s_size_bytes -= node.size
                self._add_ghost(node.id, node.size)
                del self.mapping[node.id]
                return node.id

            # Should not happen, but keep safe behavior.
            if self.mapping:
                any_id, node = next(iter(self.mapping.items()))
                self._unlink(node)
                if node.q_type == "S":
                    self.s_size_bytes -= node.size
                else:
                    self.m_size_bytes -= node.size
                del self.mapping[any_id]
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
        self.s_size_bytes = 0
        self.m_size_bytes = 0
        self.ghost_size = 0


def init_hook(params: CommonCacheParams):
    return S3FIFO(cache_size=int(params.cache_size))


def hit_hook(data: S3FIFO, req: Request):
    data.on_hit(req)


def miss_hook(data: S3FIFO, req: Request):
    data.on_miss(req)


def eviction_hook(data: S3FIFO, req: Request):
    return data.pick_victim(req)


def remove_hook(data: S3FIFO, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: S3FIFO):
    data.on_free()
