# esbmc/esbmc#7872 spelled with a bare `list`: the non-scalar member is a plain
# name rather than a subscript, so it reaches a different arm of the widening.
def foo(y: list) -> int | list:
    return y


M = [1, 2, 3]

assert foo(M) == M
