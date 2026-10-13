import math


def positive(x: float) -> bool:
    return math.isfinite(x) and int(x) > 0


assert not positive(float("nan"))
assert not positive(float("inf"))
assert positive(2.5)

xs: list[float] = [0.5, 2.5, 3.5]
n: int = 0
while int(xs.pop()) > 0:
    n += 1
assert n == 2

inf: float = float("inf")
chained: bool = 1 < 0 < int(inf)
assert not chained

m: int = 0
while m > 0 and int(inf) > 0:
    m -= 1
assert m == 0
