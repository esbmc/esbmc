x = 42
try:
    x()
    assert False
except TypeError:
    pass
