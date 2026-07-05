def main() -> None:
    # union/intersection/difference are variadic in CPython (s.method(*others)).
    # They previously raised "<m>() takes exactly one argument"; they now fold
    # left over the arguments, and the zero-argument form returns a copy.
    # (Element/len checks keep the proof small; receivers are variables.)
    a = {1}
    u = a.union({2}, {3})
    assert 1 in u and 2 in u and 3 in u and len(u) == 3

    # Zero-argument form: a fresh copy of the set.
    c = {7, 8}
    cc = c.union()
    assert len(cc) == 2 and 7 in cc


main()
