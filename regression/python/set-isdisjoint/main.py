def main() -> None:
    a = {1, 2, 3}
    b = {4, 5, 6}
    # Disjoint: no shared elements.
    assert a.isdisjoint(b)
    # Shares 3, so not disjoint.
    c = {3, 4}
    assert not a.isdisjoint(c)
    # Empty set is disjoint with everything.
    e = set()
    assert e.isdisjoint(a)
    # set(<iterable>) receiver fast path.
    assert set([7, 8]).isdisjoint(set([9, 10]))
main()
