
def main() -> None:
    l = nondet_list(nondet_int)
    if len(l) > 4:
        return
    i = nondet_int()
    n = len(l)
    if not 0 <= i <= n:
        return
    r = l[:i]
    r[i:] = reversed(l[i:])
    assert len(r) == n
    k = 0
    while k < i:
        assert r[k] == l[k]
        k += 1
    while k < n:
        assert r[k] == l[n - 1 - k + i]
        k += 1


main()
