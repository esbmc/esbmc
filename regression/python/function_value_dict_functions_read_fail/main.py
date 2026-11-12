# As function_value_dict_functions_read, but the "g" path asserts what is false.
def f():
    return 1


def g():
    return 2


d = {"f": f, "g": g}
for k in ("f", "g"):
    v = d[k]
    assert v() == 1
