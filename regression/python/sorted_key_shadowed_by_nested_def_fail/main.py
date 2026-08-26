def second(p) -> int:
    return p[1]


def run():
    def second(p) -> int:
        return -p[1]

    xs = [(1, 9), (2, 1)]
    ks = sorted(xs, key=second)
    u, v = ks[0]
    assert u == 2


run()
