def half(n: int) -> int:
    return n // 2


def scale(n: int) -> float:
    return n / 2.5 + n % -3 + n / float("nan") + n // True


def desc(xs: list[int]) -> list[int]:
    return sorted(xs, reverse=True)


x: int = half(7) % 3
assert x == 0
assert scale(4) != 0.0
assert desc([1, 3, 2])[0] == 3
