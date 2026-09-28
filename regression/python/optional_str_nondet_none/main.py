from typing import Optional


def f(s: int) -> Optional[str]:
    if s == 200:
        return "OK"
    return None


r: Optional[str] = f(nondet_int())
if r is not None:
    assert r == "OK"
