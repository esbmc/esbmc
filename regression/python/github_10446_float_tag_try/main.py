d = {}
d["a"] = "s"
d["a"] = 1.5
caught = False
try:
    if d["a"] > 1:
        pass
except TypeError:
    caught = True
assert not caught
