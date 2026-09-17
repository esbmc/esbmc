# A call that omits a parameter with a container default must receive that
# container, not None. The preprocessor used to substitute None, so the
# argument arrived as a null pointer and len() dereferenced it.


class Node:
    def __init__(self, value, successors=[]):
        self.value = value
        self.successors = successors


def sizes(xs=[], m={}, s=set()):
    return len(xs) + len(m) + len(s)


n1 = Node(1)
assert len(n1.successors) == 0

n2 = Node(2, [n1])
assert len(n2.successors) == 1
assert len(n1.successors) == 0

assert sizes() == 0
assert sizes([1, 2]) == 2
