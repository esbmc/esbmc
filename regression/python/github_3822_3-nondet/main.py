# Nondet: instance attribute set via function should be visible in caller
class Box:
    size = 0

def resize(b, v: int):
    b.size = v

b = Box()
n: int = nondet_int()
__ESBMC_assume(n > 0 and n < 100)
resize(b, n)
assert b.size == n
assert b.size > 0
