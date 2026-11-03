class Node:

    def __init__(self, value: int) -> None:
        self.value = value


def get(node_heap, wanted_node):
    for dist, node in node_heap:
        if node is wanted_node:
            return dist
    return 0


a = Node(1)
heap = [(7, a)]
assert get(heap, a) == 0
