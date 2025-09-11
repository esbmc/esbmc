a = b = 1.0
assert a == 1.0
assert b == 1.0
a += 2.0
b *= 2
assert a == 3.0
assert b == 2.0

flag1 = flag2 = None
assert flag1 is None
assert flag2 is None


def f():
    return 42


m = n = f()
assert m == 42
assert n == 42
m += 1
n -= 1
assert m == 43
assert n == 41
