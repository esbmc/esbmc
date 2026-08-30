def run():
    xs = [[3, 1], [1, 9, 9]]
    ys = xs[0]

    xs[0] = xs[1]
    assert len(xs[0]) == 3
    assert xs[0][2] == 9

    # The overwritten element is still reachable through its own binding.
    assert len(ys) == 2

    xs[0], xs[1] = xs[1], xs[0]
    assert len(xs[1]) == 3

    xs[0] = [7, 7, 7, 7]
    assert len(xs[0]) == 4


run()
