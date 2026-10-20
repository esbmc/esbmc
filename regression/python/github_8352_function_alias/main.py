# A variable bound to a function, not a class, still calls it (#8352).
def g() -> int:
    return 2


f = g
assert f() == 2
