from typing import Tuple


def make_pair() -> Tuple[int, int]:
    return (3, 4)


x: int
y: int
x, y = make_pair()
assert x == 3
assert y == 4

x *= 2  # still int
y *= 3  # still int
assert x == 6
assert y == 12
