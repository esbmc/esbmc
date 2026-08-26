def first(p) -> int:
    return p[0]


def run():
    xs = [(3, 4), (1, 2)]
    ks = sorted(xs, key=first)
    u, v = ks[0]
    assert u == 1


run()
