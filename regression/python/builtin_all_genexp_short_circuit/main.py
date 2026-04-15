side_calls: int = 0


def side(y: int) -> bool:
    global side_calls
    side_calls = side_calls + 1
    return y > 0


xs: list[int] = [0, 1]
ys: list[int] = [1, 2]
r: bool = all(x > 0 for x in xs for y in ys if side(y))

assert r == False
assert side_calls == 1
