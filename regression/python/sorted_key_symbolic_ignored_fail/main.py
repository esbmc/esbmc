def neg(x: int) -> int:
    return -x


def run(a: int):
    xs = [a, a + 1, a + 2]
    # a is what a key-ignoring sort puts first; applying key=neg puts a + 2 there.
    assert sorted(xs, key=neg)[0] == a


run(3)
