from typing import Optional


def f(s: int) -> Optional[str]:
    if s == 200:
        return "OK"
    return None


r = f(nondet_int())
assert r is not None
