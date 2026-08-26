def second(p) -> int:
    return p[1]


if True:

    def second(p) -> int:
        return p[0]


def run():
    xs = [(1, 9), (2, 1)]
    ks = sorted(xs, key=second)
    u, v = ks[0]
    assert u == 1


run()
