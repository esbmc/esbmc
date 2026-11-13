def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 2:
        return
    halves = [x / 2 for x in arr if x > 0]
    assert len(halves) <= len(arr)
    if halves:
        assert halves[0] > 0


main()
