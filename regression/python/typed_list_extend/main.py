def main() -> None:
    n = nondet_int()
    x = nondet_int()
    if n < 0 or n > 3:
        return
    a = [1, 2]
    a.extend([x] * n)
    assert len(a) == 2 + n
    assert a[0] == 1 and a[1] == 2
    i = 2
    while i < len(a):
        assert a[i] == x
        i += 1
    b = [n]
    a.extend(b)
    assert a[2 + n] == n
    a.extend(a)
    assert len(a) == 6 + 2 * n
    assert a[3 + n] == 1 and a[5 + 2 * n] == n


main()
