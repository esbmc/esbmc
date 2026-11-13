def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    below = [x for x in arr if x < 0]
    above = [x - 1 for x in arr[1:] if x > 0]
    assert len(below) + len(above) <= len(arr)
    for x in below:
        assert x < 0
    for x in above:
        assert x >= 0


main()
