def third(p) -> int:
    return p[2]


def run():
    xs = [(1, 9), (2, 1)]
    k = third(xs[0])
    assert k == 0


run()
