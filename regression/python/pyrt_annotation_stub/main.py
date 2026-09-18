# A `...` body is a declaration stub, not code: its implicit None must not be
# reported against the return annotation.
def nondet() -> int: ...


x: int = 1
assert x == 1
