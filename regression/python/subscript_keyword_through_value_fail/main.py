# The call gives 2, not the default 1.
def g(x=1):
    return x


d = {"g": g}
assert d["g"](x=2) == 1
