# A generator expression passed straight to `extend` used to reach the C++
# frontend unlowered, so __ESBMC_values_equal compared against an invalid
# pointer. Lowering it like any other generator expression (matching
# bytes(genexp), etc.) gives extend a real list, and this now verifies.
from collections import deque


class Node:
    def __init__(self, value: int, successors):
        self.value = value
        self.successors = successors


def main():
    a = Node(1, [])
    b = Node(2, [a])

    queue = deque()
    queue.append(b)
    seen = set()
    seen.add(b)

    node = queue.popleft()
    queue.extend(s for s in node.successors if s not in seen)

    assert len(queue) == 1


main()
