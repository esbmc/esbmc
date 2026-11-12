import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) != 2:
        return
    copy = list(arr)
    assert (copy[0] < copy[1]) == (arr[0] < arr[1])
    rev = reversed(arr)
    for x in rev:
        assert x <= arr[0] or x <= arr[1]


main()
