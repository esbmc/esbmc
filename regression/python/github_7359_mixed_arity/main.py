def second(p) -> int:
    return p[1]


def run():
    a = second((1, 9))
    b = second((5, 9, 3))
    assert a == 9
    assert b == 9


run()
