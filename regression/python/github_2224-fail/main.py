x = 90

while True:
    n = nondet_int()
    __ESBMC_assume(n == 0 or n == 1)

    if n == 0 and x <= 100:
        x = x + 1
    elif n == 1 and x > 0:
        x = x - 1
    else:
        pass
    assert 0 <= x <= 100
