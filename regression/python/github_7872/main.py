# esbmc/esbmc#7872: a union return type mixing a scalar with a non-scalar
# narrowed to the scalar, so the call's result compared unequal to the list it
# returned and the assertion folded to false.
def foo(y: list[int]) -> int | list[int]:
    return y


N = nondet_int()
M = []
for i in range(N):
    M += [nondet_int()]

assert foo(M) == M
