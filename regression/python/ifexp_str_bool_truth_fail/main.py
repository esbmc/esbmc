def f(c: bool, s: str) -> int:
    if (s if c else False):
        return 1
    return 0


assert f(True, "a") == 1
assert f(True, "") == 1
