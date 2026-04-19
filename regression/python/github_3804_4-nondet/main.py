# reversed(range(n)) with nondet bound
def nondet_int() -> int: ...

n: int = nondet_int()
__ESBMC_assume(n >= 1)
__ESBMC_assume(n <= 5)

# Last value in reversed(range(n)) is always 0
last: int = -1
for i in reversed(range(n)):
    last = i
assert last == 0
