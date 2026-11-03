def apply(a):
    return a(1)


def chain(a):
    return apply(a)


if nondet_bool():
    last = lambda x: x + 3
else:
    last = lambda x: str(x)
d = chain(last)
e = d + 1
