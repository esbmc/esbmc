# As github_8970_dict_returned_read, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}


def pick(k):
    return d[k]()


c = nondet_bool()
if c:
    k = "a"
else:
    k = "b"
r = pick(k)
if not c:
    assert r != 1
