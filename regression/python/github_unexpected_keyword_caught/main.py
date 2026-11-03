def inner(v):
    return v


try:
    inner(v=1, x=2)
    assert False
except TypeError:
    pass
