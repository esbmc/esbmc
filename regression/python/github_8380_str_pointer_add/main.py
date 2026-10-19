class A:
    def get(self) -> str:
        return "x"


def f(x):
    return x + 1


def g(s: str) -> str:
    return s + "y"


try:
    y = A().get() + 1
    assert False
except TypeError:
    pass
try:
    y = 1 + A().get()
    assert False
except TypeError:
    pass
assert f(2) == 3
assert g("x") == "xy"
assert A().get() + "z" == "xz"
