def f(xs):
    return xs


l = f(f([4, 5]))
assert l[1] == 5
assert f(f([4, 5]))[1] == 5
