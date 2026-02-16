def test_list_repetition() -> None:
    n: int = nondet_int()
    __ESBMC_assume(n >= 0)
    __ESBMC_assume(n <= 2)

    data: list[int] = [1] * n


test_list_repetition()
