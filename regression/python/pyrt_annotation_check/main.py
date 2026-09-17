x: int = 5
s: str = "ab"


def f(a: int, b: str) -> int:
    return a + len(b)


assert x == 5
assert s == "ab"
assert f(1, "xy") == 3
