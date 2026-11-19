# 9829: a function reads a global that several module bindings may bind
def g() -> int:
    return a


c = nondet_bool()
if c:
    a: int = 1
if not c:
    a: int = 2
assert g() > 0
