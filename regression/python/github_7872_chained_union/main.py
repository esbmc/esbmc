# esbmc/esbmc#7872 in a chained union. `|` is left-associative, so the
# non-scalar sits inside the left BinOp and is only seen by recursing into it.
def foo(y: list[int]) -> int | list[int] | bool:
    return y


M = [1, 2, 3]

assert foo(M) == M
