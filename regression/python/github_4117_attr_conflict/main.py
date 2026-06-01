# Flow-sensitive class tracking (#4771): when an attribute is assigned values
# of different classes at unconditional top-level scope, the usage-site scanner
# still refuses to commit a single static type (the field stays any_type()), but
# flow_class_map_ records the LAST write per lvalue and the nested-attribute
# read casts the base to that class. So `n1.next` resolves to the Other instance
# last assigned, matching CPython, instead of aborting on the any_type() field.
# The tracking is gated to straight-line depth-1 code and cleared across
# control-flow joins, so a class is never adopted unsoundly.

class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


class Other:
    # Keeps a compatible `.value` field so the Python program runs; the
    # point of the test is the *static* type conflict seen by the
    # scanner, not a Python-level attribute error.
    def __init__(self, v):
        self.value = v


n1 = Node(1)
n2 = Node(2)
o = Other(99)

n1.next = n2
n1.next = o     # conflict: scanner sees Node*, then Other* — gives up

# CPython runtime: n1.next is `o`, so n1.next.value == 99.
# ESBMC ideal: verify SUCCESSFUL against that runtime.
# ESBMC today: .next stays any_type(), nested access aborts.
assert n1.next.value == 99
