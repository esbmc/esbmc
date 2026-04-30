def test() -> None:
    t = (1, 2, 3)
    v = (1,)
    e = {}
    e[t] = 1
    e[v] = 2
    assert e[t] == 1
    assert e[v] == 2
    assert e == {(1, 2, 3): 1, (1,): 2}


test()
