d = {}
try:
    del d["x"]
    assert False
except KeyError:
    assert True

