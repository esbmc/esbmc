from typing import Optional


def f(s: int) -> Optional["Node"]:
    if s == 200:
        return Node(1)
    return None


class Node:
    def __init__(self, v: int):
        self.v = v


r = f(nondet_int())
if r is not None:
    assert r.v == 1
