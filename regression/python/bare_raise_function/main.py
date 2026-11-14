def g() -> None:
    d = {1: 2}
    y = d[3]


def f() -> None:
    try:
        x = [1, 2][5]
    except IndexError:
        try:
            g()
        except KeyError:
            pass
        raise


try:
    f()
except KeyError:
    assert False
except IndexError:
    pass
