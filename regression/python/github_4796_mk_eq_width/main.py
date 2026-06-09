# Regression for #4796: reversing a linked list and comparing the result to a
# node with `==` made the frontend emit an equality between a Node object
# pointer (64-bit) and a None modelled at width 0. The per-solver mk_eq asserted
# equal operand widths, but that assert is elided under NDEBUG, so release builds
# fed mismatched terms to the solver API and crashed with a raw SIGSEGV.
#
# The solver backends now route through smt_convt::check_eq_operand_widths,
# which stops with a clear diagnostic in every build configuration instead of
# crashing. This test pins that behaviour: ESBMC must emit the bit-width
# mismatch error (and abort) rather than segfault. The underlying #4796 codegen
# bug (None typed at a mismatched width) is tracked separately under the
# object/Optional model rework (#4653/#3067).
from node import Node


def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode = node
        node = nextnode
    return prevnode


def test1():
    node1 = Node(1)
    node2 = Node(2, node1)
    result = reverse_linked_list(node2)
    assert result == node1


def test2():
    node = Node(0)
    result = reverse_linked_list(node)
    assert result == node


test2()
