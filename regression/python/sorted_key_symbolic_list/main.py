def neg(x: int) -> int:
    return -x


def run(a: int):
    xs = [a, a + 1, a + 2]
    ks = sorted(xs, key=neg)
    assert ks[0] == a + 2
    assert ks[2] == a
    assert xs[0] == a


run(3)
