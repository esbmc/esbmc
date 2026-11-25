# The callee stores into the element it is given, so the caller's display
# does not type it (#8914).
def f(n, items):
    items[0] = [7.5, 1]
    w, v = items[0]
    assert w > 100


args = [100, [[60, 10]]]
f(args[0], args[1])
