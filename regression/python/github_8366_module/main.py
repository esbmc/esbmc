import mod

m = mod
assert m.func1() == 42


def g() -> str:
    return m.key


assert g() == "a"
