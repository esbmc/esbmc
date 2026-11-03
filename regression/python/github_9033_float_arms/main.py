def f() -> str:
    return "a"


def h() -> float:
    return 42.5


c = nondet_bool()
r = h() if c else f()
assert isinstance(r, float) == c
assert isinstance(r, str) != c
