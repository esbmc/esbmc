# The parameter has no single element type across its call sites, so
# the unpacking is refused.
def f(xs):
    [a, b], [p, q] = xs
    return b


assert f([[1, 2], [3, 4]]) == 2
assert f([["a", "b"], ["c", "d"]]) == "a"
