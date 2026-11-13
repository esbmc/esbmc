def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4:
        return
    s = sorted(arr, reverse=True)
    i = 0
    while i + 1 < len(s):
        assert s[i] <= s[i + 1]
        i += 1


main()
