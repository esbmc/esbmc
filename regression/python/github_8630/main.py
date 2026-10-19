def f(a):
    return a + 1


def g(a):
    return a + 1


xs = nondet_list(nondet_int)
c = [f(a) for a in [g(b) for b in xs]]
for v in c:
    assert isinstance(v, int)
