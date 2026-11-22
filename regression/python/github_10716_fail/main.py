# xs[len(xs)] raises an IndexError that except IndexError catches (#10716).
xs = [1, 2]
try:
    v = xs[len(xs)]
except IndexError:
    v = 0
assert v == 1
