
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
    """Case 1: 5-node list input
    Expected Output: 1 2 3 4 5
    """

    node1 = Node(1)
    node2 = Node(2, node1)
    node3 = Node(3, node2)
    node4 = Node(4, node3)
    node5 = Node(5, node4)

    result = reverse_linked_list(node5)
    assert result == node1

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert output == [1, 2, 3, 4, 5]


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