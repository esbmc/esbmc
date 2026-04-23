# Known limitation of the usage-site scanner: instance variables created
# inside function bodies are not tracked, so `<var>.attr = <other>`
# assignments happening inside a function don't refine the attribute's
# inferred type. The fix would need scope-aware tracking of local
# instance variables.

class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


def setup() -> Node:
    n1 = Node(1)
    n2 = Node(2)
    n1.next = n2
    return n1


n = setup()

assert n.next.value == 2
