def m(v, args) -> bool:
    return any(isinstance(v, a) for a in args)


assert m(1, (str, int))
assert m(1.5, (str, int))
