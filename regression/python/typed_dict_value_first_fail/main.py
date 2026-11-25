def main() -> None:
    k = nondet_int()
    d = {}
    try:
        d[k] = d[k] + 1
    except KeyError:
        pass
    v = d[k]


main()
