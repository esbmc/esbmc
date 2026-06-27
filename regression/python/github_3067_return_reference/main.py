# A function returning a user-defined class instance returns a *reference*
# (pointer) to a heap object, matching CPython. The returned object outlives
# the callee frame and preserves identity/aliasing across the call boundary.
class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next = None


def make(v: int) -> Node:
    # Constructor used directly as a return value: must escape to the heap.
    return Node(v)


def identity(n: Node) -> Node:
    # Returning a parameter must return the same object, not a copy.
    return n


a = make(7)
b = a                 # aliasing: two names, one object
b.value = 100
assert a.value == 100  # mutation through b is visible through a
assert a is b

c = identity(a)
c.value = 55
assert a.value == 55   # c aliases a
assert c is a

# A returned object can link to another returned object and survive the frame.
head = make(1)
head.next = make(2)
assert head.next.value == 2
