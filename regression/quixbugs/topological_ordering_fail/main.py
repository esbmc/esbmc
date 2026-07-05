# KNOWNBUG (#4805). The QuixBugs `topological_ordering` bug is that the guard
# reads `nextnode.outgoing_nodes` where it should read `nextnode.incoming_nodes`.
# ESBMC cannot genuinely detect it within a practical BMC budget: the algorithm
# builds a list of graph nodes by comprehension and then repeatedly evaluates
# `set(ordered_nodes).issuperset(...)`, `nextnode not in ordered_nodes`, and
# `ordered_nodes.append(...)` over a symbolic-size list of heap objects. That
# membership/append machinery is combinatorially hard for the SMT backend -- it
# does not converge even on a 3-node graph (decision procedure ~50s at one
# unwind; full graphs time out). This is the #5121 symbolic-lists-of-objects
# scalability wall, the same wall as the sibling `github_4782_object_size`.
# The object-handling correctness fixes that #4805 surfaced have all merged
# (issuperset lowering / object_size abort / OM copy in PR #5306; Class*-pointer
# reads in PR #5339; `obj in [obj]` membership in PR #5569; list-comprehension
# element type-id preservation, with `github_4805_comprehension_obj`). What
# remains is pure solver scalability, so the test stays KNOWNBUG. The earlier
# `VERIFICATION FAILED` here was vacuous -- it came from a truncated-loop
# unwinding assertion, not from detecting the algorithmic bug (adding
# `--no-unwinding-assertions` makes ESBMC report SUCCESSFUL). The KNOWNBUG flags
# keep `--no-unwinding-assertions` so the recorded result honestly reflects that
# ESBMC cannot detect the bug, mirroring the sibling `topological_ordering`.
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

def topological_ordering(nodes):
    ordered_nodes = [node for node in nodes if not node.incoming_nodes]

    for node in ordered_nodes:
        for nextnode in node.outgoing_nodes:
            if set(ordered_nodes).issuperset(nextnode.outgoing_nodes) and nextnode not in ordered_nodes:
                ordered_nodes.append(nextnode)

    return ordered_nodes

"""
Topological Sort

Input:
    nodes: A list of directed graph nodes

Precondition:
    The input graph is acyclic

Output:
    An OrderedSet containing the elements of nodes in an order that puts each node before all the nodes it has edges to
"""

def test1():
    """Case 1: Wikipedia graph
    Output: 5 7 3 11 8 10 2 9
    """
    five = Node(5)
    seven = Node(7)
    three = Node(3)
    eleven = Node(11)
    eight = Node(8)
    two = Node(2)
    nine = Node(9)
    ten = Node(10)
    five.outgoing_nodes = [eleven]
    seven.outgoing_nodes = [eleven, eight]
    three.outgoing_nodes = [eight, ten]
    eleven.incoming_nodes = [five, seven]
    eleven.outgoing_nodes = [two, nine, ten]
    eight.incoming_nodes = [seven, three]
    eight.outgoing_nodes = [nine]
    two.incoming_nodes = [eleven]
    nine.incoming_nodes = [eleven, eight]
    ten.incoming_nodes = [eleven, three]
    result = [x.value for x in topological_ordering([five, seven, three, eleven, eight, two, nine, ten])]
    assert result == [5, 7, 3, 11, 8, 10, 2, 9]

def test2():
    """Case 2: GeekforGeeks example
    Output: 4 5 0 2 3 1
    """
    five = Node(5)
    zero = Node(0)
    four = Node(4)
    one = Node(1)
    two = Node(2)
    three = Node(3)
    five.outgoing_nodes = [two, zero]
    four.outgoing_nodes = [zero, one]
    two.incoming_nodes = [five]
    two.outgoing_nodes = [three]
    zero.incoming_nodes = [five, four]
    one.incoming_nodes = [four, three]
    three.incoming_nodes = [two]
    three.outgoing_nodes = [one]
    result = [x.value for x in topological_ordering([zero, one, two, three, four, five])]
    assert result == [4, 5, 0, 2, 3, 1]

def test3():
    """Case 3: Cooking with InteractivePython"""
    milk = Node('3/4 cup milk')
    egg = Node('1 egg')
    oil = Node('1 Tbl oil')
    mix = Node('1 cup mix')
    syrup = Node('heat syrup')
    griddle = Node('heat griddle')
    pour = Node('pour 1/4 cup')
    turn = Node('turn when bubbly')
    eat = Node('eat')
    milk.outgoing_nodes = [mix]
    egg.outgoing_nodes = [mix]
    oil.outgoing_nodes = [mix]
    mix.incoming_nodes = [milk, egg, oil]
    mix.outgoing_nodes = [syrup, pour]
    griddle.outgoing_nodes = [pour]
    pour.incoming_nodes = [mix, griddle]
    pour.outgoing_nodes = [turn]
    turn.incoming_nodes = [pour]
    turn.outgoing_nodes = [eat]
    syrup.incoming_nodes = [mix]
    syrup.outgoing_nodes = [eat]
    eat.incoming_nodes = [syrup, turn]
    result = [x.value for x in topological_ordering([milk, egg, oil, mix, syrup, griddle, pour, turn, eat])]
    expected = ['3/4 cup milk', '1 egg', '1 Tbl oil', 'heat griddle', '1 cup mix', 'pour 1/4 cup', 'heat syrup', 'turn when bubbly', 'eat']
    assert result == expected

test1()
test2()
test3()
