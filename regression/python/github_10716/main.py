# xs[len(xs)] raises an IndexError that except IndexError catches (#10716).
xs = [1, 2]
try:
    v = xs[len(xs)]
    assert False
except IndexError:
    pass

ys = [3, 4, 5]
assert ys[len(xs)] == 5
