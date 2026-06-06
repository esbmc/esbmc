# List repetition [x] * n must produce n elements when the count is a runtime
# value -- a bare symbol, a compound expression (m + 1), or the count on the
# left (n * [x]). Regression for issue #5146 (frq = [0] * (max(lst) + 1)).
from typing import List


def repeat_sym(n: int) -> List[int]:
    return [0] * n


def repeat_expr(m: int) -> int:
    a = [0] * (m + 1)
    return len(a)


def repeat_left(n: int) -> List[int]:
    return n * [7]


def main() -> None:
    a = repeat_sym(6)
    assert len(a) == 6
    assert a[0] == 0
    assert a[5] == 0

    assert repeat_expr(5) == 6

    b = repeat_left(4)
    assert len(b) == 4
    s = 0
    for x in b:
        s += x
    assert s == 28


main()
