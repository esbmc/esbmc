l = [1, 2]
try:
    try:
        x = l[5]
    except IndexError:
        def g() -> None:
            raise
        try:
            y = {1: 2}[3]
        except KeyError:
            pass
        raise
except IndexError:
    pass
except KeyError:
    assert False
