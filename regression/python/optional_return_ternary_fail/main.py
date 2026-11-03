from typing import Optional


def f(s: int) -> Optional[str]:
    return "OK" if s == 200 else None


assert f(nondet_int()) is not None
