import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 3:
        return
    b = arr + [7]
    for i in range(len(arr)):
        assert b[i] == arr[i]


main()
