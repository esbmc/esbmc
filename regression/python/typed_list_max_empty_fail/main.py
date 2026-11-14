def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    m = min(arr)
    assert m == m


main()
