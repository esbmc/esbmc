# esbmc/esbmc#7876: a union parameter annotation collapsed to its leftmost
# member, so --strict-types rejected a list argument to `int | list[int]`.
def foo(y: int | list[int]) -> list[int]:
    return y


M = [1, 2, 3]

assert foo(M) == M
