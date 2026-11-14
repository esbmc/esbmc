# Issue #10254 review: 2.5000000000001 is above the tie, so it rounds up.
def r(x: float) -> int:
    return round(x)


assert r(2.5000000000001) == 2
