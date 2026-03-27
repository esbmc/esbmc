def test_cover_reachability():
    x: int = nondet_int()

    if x > 10:
        # Cover: Can we reach this branch?
        __ESBMC_cover(True)  # Should succeed
        y: int = x * 2
    else:
        # Cover: Can we reach this branch?
        __ESBMC_cover(True)  # Should succeed
        y: int = x + 1

    # Cover: Can y be greater than 20?
    __ESBMC_cover(y > 20)  # Should succeed


test_cover_reachability()
