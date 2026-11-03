# Issue #8256: hoisting B ahead of A would run its tick() first, so it stays
# nested.
n: int = 0


def tick() -> int:
    global n
    n = n + 1
    return n


class A:
    X: int = tick()

    class B:
        Y: int = tick()


assert A.X == 1
assert A.B.Y == 2
