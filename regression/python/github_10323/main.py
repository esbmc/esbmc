import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) != 2:
        return
    s = sorted(arr)
    assert s[0] <= s[1]


main()
