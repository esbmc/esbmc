from collections import Counter


def main() -> None:
    n = nondet_int()
    if not 0 <= n <= 3:
        return
    c = Counter()
    assert not c
    for i in range(n):
        c[i, i] = c[i - 1, i - 1] + 1
    v = c[n + 1, 0]
    assert v == 0
    assert (n + 1, 0) not in c
    if n > 0:
        assert c
        assert max(c.values()) == n


main()
