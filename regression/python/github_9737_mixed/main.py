def f(c):
    q = [1] if c else ["a"]
    return len(q)


assert f(True) == 1
assert f(False) == 1
