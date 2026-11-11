# Issue #10254 review: round() of a float folds only an exact, in-range result.
def r(x: float) -> int:
    return round(x)


def ri(x: int) -> int:
    return round(x)


r(1e300)
overflowed = False
try:
    r(1e400)
except OverflowError:
    overflowed = True
assert overflowed
assert r(2.2) == 2
assert r(3.5) == 4
assert r(2.5) == 2
assert r(-3.5) == -4
assert r(2.5000000000001) == 3
assert ri(7) == 7
