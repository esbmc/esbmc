# As github_8970_dict_literal, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}
c = nondet_bool()
k = "a" if c else "b"
r = d[k]()
if not c:
    assert r != 1
