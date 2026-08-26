def neg(x: int) -> int:
    return -x


def run(a: int):
    xs = [a, a + 1]
    # key=neg reverses the order, so ks[0] is the larger value. The runtime
    # sort model has no key parameter, so this must be refused rather than
    # sorted by natural order and reported as a spurious counterexample.
    ks = sorted(xs, key=neg)
    assert ks[0] == a + 1


run(3)
