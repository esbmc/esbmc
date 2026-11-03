class Node:
    pass


a = Node()
b = Node()
assert (1, a) < (3, b)
assert (3, a) > (1, b)
assert not ((3, a) <= (1, b))
assert (1, "x", a) >= (1, "w", b)
assert (1, "a") < (2, 3)

raised = False
try:
    r = (1, a) < (1, b)
except TypeError:
    raised = True
assert raised
