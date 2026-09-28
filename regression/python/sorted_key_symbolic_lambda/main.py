def run(a: int):
    xs = [a, a + 1, a + 2]
    ks = sorted(xs, key=lambda v: -v)
    assert ks[0] == a + 2
    assert ks[2] == a


run(3)
