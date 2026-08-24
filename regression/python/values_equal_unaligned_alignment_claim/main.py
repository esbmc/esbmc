# The value pointer a generator-expression `extend` hands to
# __ESBMC_values_equal is not a valid object, so this program still fails --
# that residual defect is what keeps quixbugs/breadth_first_search KNOWNBUG
# (#4780). What this test pins is the claim that is no longer raised: an
# alignment-1 packed read cannot be misaligned, where the old uint64_t cast was
# flagged first and hid the pointer defect behind it.
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
