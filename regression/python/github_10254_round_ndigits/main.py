# round(x, n) rounds the exact binary value of x: -1.95 is just above -1.95,
# so it rounds to -1.9, and -0.445 is just below, so it rounds to -0.45.
def f(x: float, n: int) -> list[float]:
    return [round(x, n)]


assert f(-1.95, 1) == [-1.9]
assert f(-0.445, 2) == [-0.45]
