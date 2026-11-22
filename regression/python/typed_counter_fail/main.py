from collections import Counter


def main() -> None:
    n = nondet_int()
    if not 1 <= n <= 3:
        return
    c = Counter()
    for i in range(n):
        c[i, i] = c[i - 1, i - 1] + 1
    assert max(c.values()) < 3


main()
