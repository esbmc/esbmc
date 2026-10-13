def f(a):
    return a + 1


xs = [nondet_int(), nondet_int()]
c = [f(b) for b in xs]
v = c[0]
if isinstance(v, int):
    raise TypeError("t")
