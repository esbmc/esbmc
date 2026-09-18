# Counterpart to github_7872: the same union return type, with a property that
# genuinely does not hold, so the fix must not make every comparison true.
def foo(y: list[int]) -> int | list[int]:
    return y


N = nondet_int()
M = []
for i in range(N):
    M += [nondet_int()]

assert foo(M) != M
