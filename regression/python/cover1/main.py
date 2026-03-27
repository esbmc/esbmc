def test_cover_example():
    x: int = nondet_int()

    # Cover: Can x be positive?
    # Expected: FAIL (satisfiable - x can be positive)
    __ESBMC_cover(x > 0)

    # Cover: Can x be exactly 42?
    # Expected: FAIL (satisfiable - x can be 42)
    __ESBMC_cover(x == 42)

    # Add constraint
    __ESBMC_assume(x < 0)

    # Cover: Can x be positive after the assume?
    # Expected: SUCCESS (not satisfiable - x must be negative)
    __ESBMC_cover(x > 0)

    # Cover: Can x be negative?
    # Expected: FAIL (satisfiable - x can be negative)
    __ESBMC_cover(x < 0)


test_cover_example()
