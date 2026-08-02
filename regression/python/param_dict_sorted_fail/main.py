# Negative variant of param_dict_sorted: the bare-``dict`` parameter has three
# keys, so ``n_keys`` returns 3 and the assertion that it equals 2 must fail.
def n_keys(d: dict) -> int:
    ks = sorted(d)
    return len(ks)


def main() -> None:
    m = {1: 10, 2: 20, 3: 30}
    assert n_keys(m) == 2


main()
