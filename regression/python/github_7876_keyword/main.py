# esbmc/esbmc#7876 through the keyword-argument arm of the same check.
def foo(y: int | list[int]) -> list[int]:
    return y


M = [1, 2, 3]

assert foo(y=M) == M
