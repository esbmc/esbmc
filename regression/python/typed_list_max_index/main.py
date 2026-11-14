def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4:
        return
    seen = [0] * 5
    lengths = [j for j in range(1, len(arr) + 1) if arr[j - 1] > 0]
    longest = max(lengths) if lengths else 0
    seen[longest] = 1
    assert seen[longest] == 1
    assert longest <= len(arr)


main()
