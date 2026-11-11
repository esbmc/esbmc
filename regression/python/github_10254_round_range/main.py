# round() of a float beyond 64 bits is a large Python int, not a wrapped one.
def r(v: float) -> int:
    return round(v)


assert r(1e300) > 0
assert r(-1e300) < 0
