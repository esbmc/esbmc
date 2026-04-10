def f():
    return [1, 2, 3]

x = [i * 2 for i in f()]
assert x == [2, 4, 6]

