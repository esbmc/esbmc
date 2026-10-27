def f3(a, x):
    return a(x)


e = f3(lambda x: x > 0, nondet_int())
if not isinstance(e, bool):
    raise TypeError("lost")

n = nondet_int()
k = f3(lambda x: x + 1, n)
if not isinstance(k, int):
    raise TypeError("lost")
assert k == n + 1
