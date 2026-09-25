def f(*args):
    return len(args)


xs = [1, 2, 3]
assert f(*xs) == 3
