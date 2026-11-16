import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 3:
        return
    b = arr[1:]
    for i in range(len(b)):
        assert b[i] == arr[i + 1]


main()
