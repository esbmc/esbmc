def second(p) -> int:
    return p[1]


def run():
    xs = [(1, 9), (2, 1), (3, 5)]
    lo = min(xs, key=second)
    hi = max(xs, key=second)
    assert lo[0] == 2
    assert hi[0] == 1


run()
