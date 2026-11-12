def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    if arr:
        assert len(arr) > 0
        assert arr
    else:
        assert len(arr) == 0
    if not arr:
        assert len(arr) == 0
    assert bool(arr) == (len(arr) != 0)
    t = 0
    while arr:
        t = 1
        break
    assert t == (1 if arr else 0)
    empty = []
    assert not empty


main()
