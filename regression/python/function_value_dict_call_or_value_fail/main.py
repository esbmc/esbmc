# As function_value_dict_call_or_value, but the "i" path asserts what is false.
def f():
    return "a"


def g():
    return 1


d = {"s": f, "i": g, "n": 3}
for k in ("i", "n"):
    v = d[k]
    assert v == 3 or v() == 2
