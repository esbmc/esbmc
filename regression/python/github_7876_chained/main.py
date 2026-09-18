# esbmc/esbmc#7876 in a chained union. `|` is left-associative, so the list
# member sits inside the left BinOp and is only seen by walking into it.
def foo(y: int | list[int] | bool) -> list[int]:
    return y


M = [1, 2, 3]

assert foo(M) == M
