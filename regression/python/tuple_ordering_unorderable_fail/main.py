class Node:
    pass


a = Node()
b = Node()
assert (1, a) < (3, b)
r = (2, "x", a) < (2, "x", b)
