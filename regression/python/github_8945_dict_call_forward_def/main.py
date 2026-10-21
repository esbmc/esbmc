# Issue #8945: a call through a dict converted before the def it reaches.
def call(key="a"):
    return d[key]()


def f():
    return "a"


d = {"a": f}
assert call("a") == "a"
assert call() == "a"
