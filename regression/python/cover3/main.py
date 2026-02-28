def test_cover_vs_assert():
    x: int = nondet_int()
    __ESBMC_assume(0 <= x <= 100)

    # This cover succeeds: x can be 50
    __ESBMC_cover(x == 50)

    # This cover succeeds: x can be in range
    __ESBMC_cover(0 <= x <= 100)

    # This assert succeeds: x is always in range (due to assume)
    assert 0 <= x <= 100, "x in range"  # Would succeed


test_cover_vs_assert()
