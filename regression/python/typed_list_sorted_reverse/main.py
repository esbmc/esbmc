def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4:
        return
    s = sorted(arr, reverse=True)
    assert len(s) == len(arr)
    i = 0
    while i + 1 < len(s):
        assert s[i] >= s[i + 1]
        i += 1
    for x in arr:
        assert x <= s[0]


main()
