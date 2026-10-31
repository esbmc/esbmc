# Issue #9386: a negative shift count raises a catchable ValueError.
def scale(v: int, k: int) -> int:
    return v << k


n: int = nondet_int()
__ESBMC_assume(n == -1)
ok: bool = False
try:
    x: int = 8 >> n
except ValueError:
    ok = True
assert ok

k: int = nondet_int()
__ESBMC_assume(k >= 0 and k <= 4)
r: int = scale(1, k)
assert r >= 1 and r <= 16
assert (64 >> k) >= 4
