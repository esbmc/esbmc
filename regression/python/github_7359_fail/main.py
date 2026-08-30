def second(p) -> int:
    return p[1]


def run():
    xs = [(1, 9), (2, 1)]
    k = second(xs[0])
    assert k == 1


run()
