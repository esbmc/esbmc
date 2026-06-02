class Node:
    def __init__(
        self,
        value=None,
        successor=None,
        successors=[],
        predecessors=[],
        incoming_nodes=[],
        outgoing_nodes=[],
    ):
        self.value = value
        self.successor = successor
        self.successors = successors
        self.predecessors = predecessors
        self.incoming_nodes = incoming_nodes
        self.outgoing_nodes = outgoing_nodes


from collections import deque as Queue

def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while True:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            queue.extend(node for node in node.successors if node not in nodesseen)
            nodesseen.update(node.successors)

    return False



"""
Breadth-First Search


Input:
    startnode: A digraph node
    goalnode: A digraph node

Output:
    Whether goalnode is reachable from startnode
"""

def test1():
    """Case 1: Strongly connected graph
    Output: Path found!
    """
    station1 = Node(1)
    station2 = Node(2, None, [station1])
    station3 = Node(3, None, [station1, station2])
    station4 = Node(4, None, [station2, station3])
    station5 = Node(5, None, [station4, station3])
    station6 = Node(6, None, [station5, station4])
    path_found = breadth_first_search(station6, station1)
    assert path_found

def test2():
    """Case 2: Branching graph
    Output: Path found!
    """
    nodef = Node(6)
    nodee = Node(5)
    noded = Node(4)
    nodec = Node(3, None, [nodef])
    nodeb = Node(2, None, [nodee])
    nodea = Node(1, None, [nodeb, nodec, noded])
    path_found = breadth_first_search(nodea, nodee)
    assert path_found

def test3():
    """Case 3: Two unconnected nodes in graph
    Output: Path not found
    """
    nodef = Node(1)
    nodee = Node(2)
    path_found = breadth_first_search(nodef, nodee)
    assert not path_found

def test4():
    """Case 4: One node graph
    Output: Path found!
    """
    nodef = Node(1)
    path_found = breadth_first_search(nodef, nodef)
    assert path_found

def test5():
    """Case 5: Graph with cycles
    Output: Path found!
    """
    nodef = Node(6)
    nodee = Node(5)
    noded = Node(4)
    nodec = Node(3, None, [nodef])
    nodeb = Node(2, None, [nodee])
    nodea = Node(1, None, [nodeb, nodec, noded])
    nodee.successors = [nodea]
    path_found = breadth_first_search(nodea, nodef)
    assert path_found

test1()
test2()
test3()
test4()
test5()
