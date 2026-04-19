from node import Node

def detect_cycle(node):
    hare = tortoise = node

    while True:
        if hare is None or hare.successor is None:
            return False

        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True

node1 = Node(1)
node2 = Node(2, node1)
node3 = Node(3, node2)
node4 = Node(4, node3)
node5 = Node(5, node4)


def test1():
    """Case 1: 5-node list input with no cycle
    Expected Output: Cycle not detected!
    """

    detected = detect_cycle(node5)

    assert not detected


def test2():
    """Case 2: 5-node list input with cycle
    Expected Output: Cycle detected!
    """

    node1.successor = node5

    detected = detect_cycle(node5)

    assert detected


def test3():
    """Case 3: 2-node list with cycle
    Expected Output: Cycle detected!
    """

    node1.successor = node2

    detected = detect_cycle(node2)

    assert detected


def test4():
    """Case 4: 2-node list with no cycle
    Expected Output: Cycle not detected!
    """

    node6 = Node(6)
    node7 = Node(7, node6)

    detected = detect_cycle(node7)

    assert not detected


def test5():
    """Case 5: 1-node list
    Expected Output: Cycle not detected
    """

    node = Node(0)
    detected = detect_cycle(node)

    assert not detected


def test6():
    """Case 6: 5 nodes in total. the last 2 nodes form a cycle. input the first node
    Expected Output: Cycle detected!
    """

    node1.successor = node2

    detected = detect_cycle(node5)

    assert detected

test1()
test2()
test3()
test4()
test5()
test6()
