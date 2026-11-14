xs = [1, 2]
caught = False
try:
    v = xs[len(xs)]
    assert False
except IndexError:
    caught = True
assert caught

caught = False
try:
    xs[len(xs)] = 3
except IndexError:
    caught = True
assert caught

names = ["a", "b"]
caught = False
try:
    s = names[len(names)]
except IndexError:
    caught = True
assert caught

assert xs[len(xs) - 1] == 2
