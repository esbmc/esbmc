from typing import Optional


def f(s: int) -> Optional[str]:
    if s == 200:
        return "OK"
    return None


assert f(nondet_int()) is not None
