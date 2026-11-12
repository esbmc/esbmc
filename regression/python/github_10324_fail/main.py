import _sv_verifier


def main() -> None:
    arr = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
    if len(arr) > 3:
        return
    s = sorted(arr)
    total = 0
    for x in arr:
        total += x
    for x in s:
        total -= x
    assert total == 1 and len(s) == len(arr)


main()
