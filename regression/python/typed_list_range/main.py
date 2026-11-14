def main() -> None:
    n = nondet_int()
    if not 0 <= n <= 4:
        return
    xs = list(range(n))
    assert len(xs) == n
    i = 0
    while i < n:
        assert xs[i] == i
        i += 1
    assert [2, 3, 4] == list(range(2, 5))
    assert list(range(5, 2)) == []
    assert list(range(-1)) == []


main()
