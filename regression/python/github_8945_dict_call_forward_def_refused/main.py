# Issue #8945: the call is typed from the annotation, which None widens.
def call(key):
    return d[key]()


def f() -> int:
    if nondet_bool():
        return 1
    return None


d = {"a": f}
x = call("a")
assert x == 1
