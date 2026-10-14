# Regression for #4796: a bare `prevnode = None` local is modelled as a
# pointer-width integer object handle, so reverse_linked_list returns a handle
# while a freshly constructed Node is stored by value. Comparing the two with
# `==` fed a struct sort and a scalar to the solver's mk_eq, whose operand-width
# assert is elided under NDEBUG and crashed release builds with a SIGSEGV. The
# comparison now reconciles handle vs by-value instance as class-pointer
# identity, so `==`/`!=`/`is`/`is not` are well-typed and precise.
from node import Node


def reverse(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode = node
        node = nextnode
    return prevnode


def test_identity():
    n = Node(0)
    r = reverse(n)
    assert r == n
    assert r is n
    assert not (r != n)
    assert not (r is not n)


def test_distinct():
    a = Node(1)
    b = Node(2)
    r = reverse(a)
    assert r != b
    assert r is not b
    assert not (r == b)


test_identity()
test_distinct()
