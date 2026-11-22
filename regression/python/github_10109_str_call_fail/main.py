# #10109: a str-returning call stored into an Optional[str].
from typing import Optional


def f(n: int) -> str:
    if n > 0:
        return "pos"
    return "neg"


def run(n: int, want: str):
    r: Optional[str] = None
    if n < 3:
        r = f(n)
    assert r == want


run(1, "pos")
run(-1, "pos")
