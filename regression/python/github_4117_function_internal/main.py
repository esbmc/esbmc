# The usage-site scanner now tracks instance attributes written through a
# function-local variable (see github_4117_func_local_attr), so `.next`'s
# type resolves here too. This case remains KNOWNBUG for a SEPARATE reason:
# `setup()` returns `n1`, whose `.next` points to the function-local `n2`;
# ESBMC stack-allocates Python objects, so the returned reference to `n2` is
# reported as a use-after-free ("accessed expired variable pointer 'n2'").
# CPython heap-allocates objects, so this is safe at runtime. Flipping this
# test additionally needs object-lifetime (heap) modeling for objects that
# escape their defining function — orthogonal to class tracking.

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
