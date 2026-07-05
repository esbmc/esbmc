# `setup()` returns `n1`, whose `.next` points to the function-local `n2`.
# ESBMC stack-allocates Python objects, so the returned reference to `n2` used
# to be reported as a use-after-free ("accessed expired variable pointer
# 'n2'"). Python objects are garbage-collected, so an instance referenced after
# its defining function returns is valid at runtime. Symex now gives
# user-defined Python class instances whole-program (GC) lifetime, so the
# escaped reference stays valid and `n.next.value` reads the real value. See
# issue #4773.

class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


def setup() -> Node:
    n1 = Node(1)
    n2 = Node(2)
    n1.next = n2
    return n1


n = setup()

assert n.next.value == 2
