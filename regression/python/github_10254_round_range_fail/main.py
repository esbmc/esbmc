# round(1e300) is positive, so this assertion fails.
def r(v: float) -> int:
    return round(v)


assert r(1e300) < 0
