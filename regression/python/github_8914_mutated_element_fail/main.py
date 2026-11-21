# The element is replaced before the call, so its display does not type it
# (#8914). Typing it from the display would prove the assertion.
def f(n, items):
    w, v = items[0]
    assert w > 100


args = [100, [[60, 10]]]
args[1][0] = [7.5, 1]
f(args[0], args[1])
