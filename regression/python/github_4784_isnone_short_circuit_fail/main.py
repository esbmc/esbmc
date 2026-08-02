# Failing companion to github_4784_isnone_short_circuit: same looping
# short-circuit `x is None or x.attr is None` guard (which previously aborted
# the SMT backend), but with a wrong assertion so the expected verdict is
# FAILED. Pins that the fix yields a real verdict instead of crashing and that
# the verdict is sense-preserving.
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
b = Node(2, a)  # b -> a -> None, so reaches_end(b) is True
assert reaches_end(b) == False
