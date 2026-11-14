import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 1:
        return
    s = 0
    for x in arr:
        s += x
    assert len(arr) == 0 or s == arr[0]


main()
