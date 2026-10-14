# Negative counterpart of isnone_class_field: pins that `tail.nxt is None`
# genuinely evaluates to True for a None-valued Class* field, so the wrong
# expectation (has_next(tail) == 1) is a real assertion violation. Before the
# isnone-on-Class* fix this folded to False and the bug was hidden (spurious
# VERIFICATION SUCCESSFUL).

class Node:
    def __init__(self, value: int, nxt=None):
        self.value = value
        self.nxt = nxt


def has_next(n: Node) -> int:
    if n.nxt is None:
        return 0
    return 1


tail = Node(2)          # nxt defaults to None

assert has_next(tail) == 1   # WRONG: tail.nxt is None, so has_next(tail) == 0
