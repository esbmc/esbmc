t = (1, )
try:
    t[0] = 2
    assert False
except TypeError:
    pass
