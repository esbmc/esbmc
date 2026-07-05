# Negative companion to github_4796_object_handle_eq. The reconciled handle vs
# by-value comparison must be precise, not vacuously true: reversing the list
# b -> a yields head `a`, so asserting the head is `b` is genuinely false and
# ESBMC must report it rather than fold the mismatched-width compare away.
from node import Node


def reverse(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode = node
        node = nextnode
    return prevnode


def test_wrong_head():
    a = Node(1)
    b = Node(2, a)
    r = reverse(b)
    assert r == b


test_wrong_head()
