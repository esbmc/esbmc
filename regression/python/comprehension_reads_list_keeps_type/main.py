def f(a):
    return a + 1


xs = [nondet_int(), nondet_int()]
c = [f(b) for b in xs]
v = c[0]
if not isinstance(v, int):
    raise TypeError("t")
assert v == xs[0] + 1
