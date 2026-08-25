def neg(x: int) -> int:
    return -x


def run(a: int):
    xs = [a, a + 1]
    assert max(xs, key=neg) == a


run(3)
