import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    b = arr[1:]
    c = arr + arr
    assert len(b) + 1 >= len(arr)
    assert len(c) == 2 * len(arr)


main()
