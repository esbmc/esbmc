def run():
    xs = [[3, 1], [1, 9, 9], [2, 5]]
    ys = sorted(xs, key=len)

    assert len(ys[0]) == 2
    assert len(ys[1]) == 2
    assert len(ys[2]) == 3


run()
