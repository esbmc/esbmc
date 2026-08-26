def second(p) -> int:
    return p[1]


def run():
    xs = [(1, 9), (2, 1)]
    ks = sorted(xs, key=second)
    u, v = ks[0]
    assert u == 1
    assert v == 9


run()
