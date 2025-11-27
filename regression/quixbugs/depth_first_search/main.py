from node import Node

def depth_first_search(startnode, goalnode):
    nodesvisited = set()

    def search_from(node):
        if node in nodesvisited:
            return False
        elif node is goalnode:
            return True
        else:
            nodesvisited.add(node)
            return any(
                search_from(nextnode) for nextnode in node.successors
            )

    return search_from(startnode)

def test1():
    """Case 1: Strongly connected graph
    Output: Path found!
    """

    station1 = Node("Westminster")
    station2 = Node("Waterloo", None, [station1])
    station3 = Node("Trafalgar Square", None, [station1, station2])
    station4 = Node("Canary Wharf", None, [station2, station3])
    station5 = Node("London Bridge", None, [station4, station3])
    station6 = Node("Tottenham Court Road", None, [station5, station4])

    path_found = depth_first_search(station6, station1)

    assert path_found


def test2():
    """Case 2: Branching graph
    Output: Path found!
    """

    nodef = Node("F")
    nodee = Node("E")
    noded = Node("D")
    nodec = Node("C", None, [nodef])
    nodeb = Node("B", None, [nodee])
    nodea = Node("A", None, [nodeb, nodec, noded])

    path_found = depth_first_search(nodea, nodee)

    assert path_found


def test3():
    """Case 3: Two unconnected nodes in graph
    Output: Path not found
    """

    nodef = Node("F")
    nodee = Node("E")

    path_found = depth_first_search(nodef, nodee)

    assert not path_found


def test4():
    """Case 4: One node graph
    Output: Path found!
    """

    nodef = Node("F")

    path_found = depth_first_search(nodef, nodef)

    assert path_found


def test5():
    """Case 5: Graph with cycles
    Output: Path found!
    """

    nodef = Node("F")
    nodee = Node("E")
    noded = Node("D")
    nodec = Node("C", None, [nodef])
    nodeb = Node("B", None, [nodee])
    nodea = Node("A", None, [nodeb, nodec, noded])

    nodee.successors = [nodea]

    path_found = depth_first_search(nodea, nodef)

    assert path_found

test1()
test2()
test3()
test4()
test5()
