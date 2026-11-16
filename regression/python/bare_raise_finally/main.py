l = [1, 2]
d = {1: 2}
try:
    try:
        x = l[5]
    except IndexError:
        try:
            y = d[3]
        finally:
            raise
except KeyError:
    pass
except IndexError:
    assert False
