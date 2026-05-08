def main() -> None:
    t = (1, 2, 1, 3, 1)
    assert t.count(1) == 3
    assert t.count(2) == 1
    assert t.count(99) == 0

    # index returns the first match.
    u = (10, 20, 30, 20)
    assert u.index(10) == 0
    assert u.index(20) == 1
    assert u.index(30) == 2

    # bool elements compare by value too.
    b = (True, False, True)
    assert b.count(True) == 2
    assert b.index(False) == 1
main()
