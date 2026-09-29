from typing import Optional


def f(s: int) -> Optional[str]:
    return "OK" if s == 200 else None


r = f(nondet_int())
if r is not None:
    assert r == "OK"
