# The list is changed through the parameter before the unpacking, so the
# element types of the displays the call passes are not trusted.
def f(x):
    x[0][0] = "s"
    [a, b], c = x
    return a


assert f(([1, 2], 3)) == "s"
