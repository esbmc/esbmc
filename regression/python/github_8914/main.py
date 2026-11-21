# A list read out of a list mixing an int and lists keeps its element type
# when passed to an unannotated parameter, spelt out or starred (#8914).
def f(n, items):
    w, v = items[0]
    return w + v


args = [100, [[60, 10], [50, 8]]]
assert f(args[0], args[1]) == 70
assert f(*args) == 70
