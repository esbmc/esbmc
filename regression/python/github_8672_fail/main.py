def f(xs, i):
    return "v %d" % xs[i]


names = ["bob", "al"]
assert "%s and %s" % (names[0], names[1]) == "bob and al"
assert f([1, 2], 1) == "v 1"
