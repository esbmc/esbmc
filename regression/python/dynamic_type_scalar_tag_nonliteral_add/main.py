n = nondet_int()
if n > 0:
    x = n
else:
    x = "a"

if isinstance(x, int):
    y = x + 1
    assert y == n + 1
