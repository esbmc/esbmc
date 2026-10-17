def f(xs, i):
    return "v %d" % xs[i]


assert f([1, 2], 1) == "v 2"
f([1, 2], 2)
