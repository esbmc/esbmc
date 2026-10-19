def check(a: int, b: int, q: int, r: int) -> None:
    assert a // b == q
    assert a % b == r


x: int = nondet_int()
y: int = nondet_int()
__ESBMC_assume(x == -7 or x == 7)
__ESBMC_assume(y == 2 or y == -2)
if x == -7 and y == 2:
    check(x, y, -4, 1)
elif x == -7 and y == -2:
    check(x, y, 3, -1)
elif x == 7 and y == -2:
    check(x, y, -4, -1)
else:
    check(x, y, 3, 1)
