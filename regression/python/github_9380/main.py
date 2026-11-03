# Issue #9380: divmod() by zero raises a catchable ZeroDivisionError.
a: int = 7
b: int = 0
try:
    q: int = divmod(a, b)[0]
except ZeroDivisionError:
    q = 0
assert q == 0

c: float = 7.0
d: float = 0.0
try:
    r: float = divmod(c, d)[1]
except ZeroDivisionError:
    r = 1.0
assert r == 1.0

q2, r2 = divmod(-7, 2)
assert q2 == -4 and r2 == 1
x, y = divmod(7.5, 2.0)
assert x == 3.0 and y == 1.5
