from collections import Counter


def main() -> None:
    n = nondet_int()
    if not 0 <= n <= 2:
        return
    c = Counter()
    for i in range(n):
        c[i, 0] = 1
    m = max(c.values())
    assert m == 1


main()
