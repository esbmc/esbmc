# Regression for an SMT-backend abort ("Couldn't convert expression in
# unrecognised format") on a short-circuited `x is None or x.attr is None`
# guard inside a loop (QuixBugs detect_cycle / depth_first_search family).
#
# `cur.nxt` is only dereferenced when `cur is None` is false, so symex
# dereferences `cur.nxt` under the guard `not isnone(cur)`. dereference ran
# before the Python-predicate lowering, so the still-live `isnone(cur)` leaked
# into the recorded NULL-pointer dereference-safety assertion and reached the
# SMT backend, which has no conversion rule for `isnone`, aborting. Lowering
# Python predicates in the GOTO-guard before dereference fixes it.
class Node:
    def __init__(self, value, nxt=None):
        self.value = value
        self.nxt = nxt


def reaches_end(head):
    cur = head
    while True:
        if cur is None or cur.nxt is None:
            return True
        cur = cur.nxt.nxt
    return False


a = Node(1)
b = Node(2, a)  # b -> a -> None
assert reaches_end(b) == True
