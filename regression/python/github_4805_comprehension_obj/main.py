# A list comprehension over a list of class instances must preserve each
# element's class. Before the #4805 fix the comprehension stored elements with
# a zero type-id, so iterating the result and reading an attribute
# (`for x in node.outs`) dereferenced an invalid pointer. The same computation
# must now verify cleanly.


class Node:
    def __init__(self, value=0, outs=[]):
        self.value = value
        self.outs = outs


def total_out_values(nodes):
    selected = [n for n in nodes if n.value > 0]
    total = 0
    for node in selected:
        for x in node.outs:
            total = total + x.value
    return total


def test():
    a = Node(5)
    b = Node(7)
    c = Node(9)
    a.outs = [b, c]
    # b and c have no outgoing nodes, so only a contributes: 7 + 9 = 16.
    assert total_out_values([a, b, c]) == 16


test()
