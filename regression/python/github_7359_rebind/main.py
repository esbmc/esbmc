def second(p) -> int:
    return p[1]


def run():
    xs = [(1, 9)]
    xs = [[5, 6, 7]]
    k = second(xs[0])
    assert k == 6


run()
