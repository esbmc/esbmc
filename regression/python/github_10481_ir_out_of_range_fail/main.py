def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 2:
        return
    x = arr[-3]


main()
