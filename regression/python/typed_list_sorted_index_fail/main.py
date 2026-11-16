def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    assert sorted(arr)[len(arr)] >= 0


main()
