def second(p) -> int:
    q = p[1]
    return q


def run():
    xs = [(1, 9), (2, 1)]
    ks = sorted(xs, key=second)
    u, v = ks[0]
    assert u == 2


run()
