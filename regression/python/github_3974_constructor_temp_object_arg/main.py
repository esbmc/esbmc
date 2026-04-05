class Node:
    def __init__(self, successor=None):
        self.successor = successor


def mk_node():
    return Node()


x = Node(mk_node())
assert x.successor
