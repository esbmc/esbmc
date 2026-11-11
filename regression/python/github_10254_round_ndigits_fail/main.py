# round(-1.95, 1) is -1.9, not -2.0.
def f(x: float, n: int) -> list[float]:
    return [round(x, n)]


assert f(-1.95, 1) == [-2.0]
