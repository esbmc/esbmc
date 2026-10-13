# An object read out of a list keeps naming its object, so the loop bounded
# by its attribute stops.
class Node:
    def __init__(self, n: int) -> None:
        self.n = n


def count(node: Node) -> int:
    k = 0
    while k < node.n:
        k += 1
    return k


nodes = [Node(2), Node(3)]
assert count(nodes[1]) == 3
