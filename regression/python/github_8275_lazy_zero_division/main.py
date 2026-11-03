def f(d: int) -> int:
    return d


x: int = 0
ok: bool = x != 0 and 10 // x > 1
assert not ok

ok = x == 0 or 10 % x > 1
assert ok

y: float = 0.0
ok = y != 0.0 and 1.0 / y > 0.0
assert not ok

ok = 0 != x < 10 // x
assert not ok

ok = x != 0 and 10 // f(x) > 1
assert not ok

caught: bool = False
try:
    ok = x == 0 and 10 // x > 1
except ZeroDivisionError:
    caught = True
assert caught
