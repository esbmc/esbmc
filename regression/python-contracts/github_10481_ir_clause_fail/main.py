def last(l: list) -> int:
    __ESBMC_requires(len(l) > 0)
    __ESBMC_ensures(__ESBMC_return_value == l[-1])
    return l[0]


def main() -> None:
    a = nondet_list(nondet_int)
    if len(a) == 0:
        return
    v: int = last(a)
    assert v == a[-1]


main()
