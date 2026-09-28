def neg(x: int) -> int:
    return -x


def run(a: int):
    xs = [a, a + 1]
    # key=neg reverses the ordering, so the max by key is the smaller value.
    assert max(xs, key=neg) == a
    assert min(xs, key=neg) == a + 1


run(3)
