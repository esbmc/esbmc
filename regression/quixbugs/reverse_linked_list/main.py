
from node import Node

def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode = node
        node = nextnode
    return prevnode

"""
def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode, node = node, nextnode
    return prevnode

def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        node, prevnode = nextnode, node
    return prevnode

"""

def test1():
    """Case 1: 3-node list input
    Expected Output: 1 2 3

    A 3-node list fully exercises the reversal (head, middle and tail
    pointer rewiring). Larger inputs push the dynamic-list operational
    model past ESBMC's practical unwinding envelope without adding
    algorithmic coverage; test2/test3 cover the 1-node and empty cases.
    """

    node1 = Node(1)
    node2 = Node(2, node1)
    node3 = Node(3, node2)

    result = reverse_linked_list(node3)
    assert result == node1

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert output == [1, 2, 3]


def test2():
    """Case 2: 1-node list input
    Expected Output: 0
    """

    node = Node(0)

    result = reverse_linked_list(node)
    assert result == node

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert output == [0]


def test3():
    """Case 3: None input
    Expected Output: None
    """

    result = reverse_linked_list(None)
    assert not result

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert not output

test1()
test2()
test3()

