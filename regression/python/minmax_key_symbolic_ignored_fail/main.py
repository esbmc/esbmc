def neg(x: int) -> int:
    return -x


def run(a: int):
    xs = [a, a + 1]
    # a + 1 is what a key-ignoring max returns; applying key=neg gives a.
    assert max(xs, key=neg) == a + 1


run(3)
