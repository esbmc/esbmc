def main() -> None:
    a = {1, 2}
    b = {2, 3}

    # union() returns every element of both sets.
    u = a.union(b)
    assert 1 in u and 2 in u and 3 in u
    assert 4 not in u

    # intersection() keeps only shared elements.
    i = a.intersection(b)
    assert 2 in i
    assert 1 not in i

    # difference() keeps elements of a not in b; non-mutating (a unchanged).
    d = a.difference(b)
    assert 1 in d
    assert 2 not in d
    assert 3 not in a

    # frozenset receiver dispatches the same way.
    f = frozenset({7})
    assert 9 in f.union({9})


main()
