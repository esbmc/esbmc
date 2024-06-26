def factorial(n:int) -> int:
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

n:int = nondet_int()
__ESBMC_assume(n > 0);
__ESBMC_assume(n < 6);

result:int = factorial(n)
assert(result != 120)
