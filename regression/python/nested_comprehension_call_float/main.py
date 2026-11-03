# The inner call returns a float, so the outer call passes one: the
# parameter of h is not an int even though xs holds ints.
def h(a):
    return a / 2


xs = [3]
c = [h(a) for a in [h(b) for b in xs]]
for v in c:
    assert isinstance(v, float)
