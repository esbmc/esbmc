class Node:
    def __init__(self, value=0, outgoing_nodes=[]):
        self.value = value
        self.outgoing_nodes = outgoing_nodes

def total_out(n):
    total = 0
    for x in n.outgoing_nodes:
        total = total + x.value
    return total

a = Node(5)
b = Node(7)
c = Node(9)
a.outgoing_nodes = [b, c]
assert total_out(a) == 17
