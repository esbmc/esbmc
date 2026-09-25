def run():
    xs = [(1, 9), (2, 1)]
    ks = sorted(xs, key=sum)
    u, v = ks[0]
    assert u == 2


run()
