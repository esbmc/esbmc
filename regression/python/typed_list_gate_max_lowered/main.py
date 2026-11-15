def main() -> None:
    arr = nondet_list(nondet_int)
    if arr:
        assert min(arr) <= max(arr)


main()
