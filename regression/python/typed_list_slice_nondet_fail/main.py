def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4:
        return
    i = nondet_int()
    j = nondet_int()
    b = arr[i:j]
    assert len(b) < len(arr) or len(arr) == 0


main()
