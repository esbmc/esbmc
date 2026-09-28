def second(p) -> int:
    return p[1]


def run():
    xs: list[tuple[int, int]] = [(1, 9), (2, 1)]
    k = second(xs[0])
    assert k == 9


run()
