x: int = nondet_int()
y: int = x

if (nondet_bool()):
    x = x + 1
else:
    x = x + 2

assert (x != y and x == y + 1 or x == y + 2)
