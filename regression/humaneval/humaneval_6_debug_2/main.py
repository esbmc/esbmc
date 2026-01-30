# Debug 2: split inside function with constant call
from typing import List

def f(s: str) -> List[str]:
    return s.split(" ")

if __name__ == "__main__":
    xs: List[str] = f("(()()) ((()))")
    assert xs[0] == "(()())"
    assert xs[1] == "((()))"
