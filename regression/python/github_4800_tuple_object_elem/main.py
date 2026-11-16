class Node:

    def __init__(self, value: int) -> None:
        self.value = value


def get(node_heap, wanted_node):
    for dist, node in node_heap:
        if node is wanted_node:
            return dist
    return 0


def bump(node_heap):
    for dist, node in node_heap:
        node.value = node.value + dist


a = Node(1)
b = Node(2)
heap = [(5, b), (7, a)]
assert get(heap, a) == 7
assert heap[0][1] is b
bump(heap)
assert a.value == 8
