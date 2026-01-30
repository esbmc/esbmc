# Debug 3: split + simple comprehension
from typing import List

def g(s: str) -> List[str]:
    return [x for x in s.split(" ") if x]

if __name__ == "__main__":
    xs: List[str] = g("a b")
    assert xs[0] == "a"
    assert xs[1] == "b"
