import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 2:
        return
    count = 0
    for _ in arr:
        count += 1
    assert count == len(arr)


main()
