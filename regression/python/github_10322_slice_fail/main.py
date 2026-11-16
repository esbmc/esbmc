import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 3 or len(arr) < 2 or arr[0] == arr[1]:
        return
    b = arr[1:]
    for i in range(len(b)):
        assert b[i] == arr[i]


main()
