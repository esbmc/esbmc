def main() -> None:
    defs: list[int] = [nondet_int(), nondet_int()]
    while len(defs) >= 0 and defs[-1] >= 0:
        defs.pop()


main()
