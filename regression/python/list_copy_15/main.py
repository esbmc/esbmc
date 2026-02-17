def test() -> None:
    a: list[int] = [1, 2, 3]
    b: list[int] = a.copy()
    c: list[int] = b
    
    c[0] = 99
    
    # b and c point to same list
    assert b[0] == 99
    assert c[0] == 99
    
    # a is independent
    assert a[0] == 1

test()
