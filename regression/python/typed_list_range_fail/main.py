def main() -> None:
    n = nondet_int()
    if not 0 <= n <= 4:
        return
    xs = list(range(n))
    assert xs == list(range(1, n + 1))


main()
