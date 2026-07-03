# `x.field is None` must evaluate as a null-pointer check when the field holds
# a heap-allocated class instance (Class*), which is NULL exactly when its
# Python value is None. Regression for the isnone evaluator folding
# `Class* is None` to a constant False (defeating None guards and causing
# spurious NULL dereferences, e.g. QuixBugs detect_cycle).

class Node:
    def __init__(self, value: int, nxt=None):
        self.value = value
        self.nxt = nxt


def has_next(n: Node) -> int:
    if n.nxt is None:
        return 0
    return 1


tail = Node(2)          # nxt defaults to None
head = Node(1, tail)    # nxt is a real Node -> field typed Node*

assert has_next(head) == 1   # head.nxt is not None
assert has_next(tail) == 0   # tail.nxt is None
