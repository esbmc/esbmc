def test() -> None:
    # setdefault stores the list value as a raw PyListObject* (size==0
    # layout). Must compare equal to a literal dict where the same list
    # lives via pointer-to-PyListObject* (size>0 layout).
    a = {}
    a.setdefault(1, []).append(1.0)
    assert a == {1: [1.0]}

    # None values are stored as (value=NULL, size=0). They must not be
    # treated as a setdefault list pointer.
    assert dict.fromkeys([1, 2, 3]) == {1: None, 2: None, 3: None}


test()
