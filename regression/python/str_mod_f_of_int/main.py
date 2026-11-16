# %f converts an int or a bool with float() first, as CPython does.
def f(y: int) -> str:
    return "%f" % y


def g(b: bool) -> str:
    return "%f" % b


assert f(3) == "3.000000"
assert f(-2) == "-2.000000"
assert g(True) == "1.000000"
