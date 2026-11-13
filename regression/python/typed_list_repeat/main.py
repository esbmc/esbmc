def main() -> None:
    n = nondet_int()
    x = nondet_int()
    if n > 4:
        return
    a = [x] * n
    b = (n - 1) * [5]
    assert len(a) == (n if n > 0 else 0)
    assert len(b) == (n - 1 if n > 1 else 0)
    i = 0
    while i < len(a):
        assert a[i] == x
        i += 1
    i = 0
    while i < len(b):
        assert b[i] == 5
        i += 1


main()
