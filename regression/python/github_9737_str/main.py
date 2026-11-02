def f(c):
    q = ["a"] if c else ["b", "c"]
    return q


a = f(True)
b = f(False)
assert isinstance(a[0], str)
assert b[1] == "c"
