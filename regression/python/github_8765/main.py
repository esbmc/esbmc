def m(v, args) -> bool:
    return any(isinstance(v, a) for a in args)


ts = (str, int)
n = 0
for a in ts:
    if a is int:
        n += 1
assert n == 1
assert m(1, (str, int))
assert not m(1.5, (str, int))
