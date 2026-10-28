flag = nondet_bool()
x: int = 1 if flag else 2
assert isinstance(x, int)
assert x == 1 or x == 2
