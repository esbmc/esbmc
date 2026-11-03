def f():
    return 1


b = ["s"]
b[0] = f
r = b[0]()
assert r == 2
