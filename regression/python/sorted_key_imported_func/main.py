from keyfuncs import value_of


def run():
    xs = [(1, 30), (2, 10)]
    ps = sorted(xs, key=value_of)
    assert ps[0][0] == 2


run()
