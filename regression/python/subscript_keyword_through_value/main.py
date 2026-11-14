# The keyword argument reaches g through the dict, so the call gives 2.
def g(x=1):
    return x


d = {"g": g}
assert d["g"](x=2) == 2
