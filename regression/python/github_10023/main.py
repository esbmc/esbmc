import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 3:
        return
    b = []
    for x in arr:
        b.append(x)
    assert b == arr


main()
