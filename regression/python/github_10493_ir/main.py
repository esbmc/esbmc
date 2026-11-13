def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 3:
        return
    if arr[0] >= 0 or arr[1] >= 0:
        return
    assert arr[1] in arr
    assert arr.count(arr[1]) >= 1
    assert arr.index(arr[0]) == 0
    x = arr[1]
    arr.remove(arr[0])
    assert arr[0] == x


main()
