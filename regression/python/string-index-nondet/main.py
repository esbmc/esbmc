def main() -> None:
    s = nondet_str()
    __ESBMC_assume(len(s) > 0)
    __ESBMC_assume(s[0] == "a")

    assert s.index("a") == 0


main()
