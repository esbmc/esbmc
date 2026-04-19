# Test nested list comparison with configurable depth
def test() -> None:
    a: list[list[list[int]]] = [[[1]]]
    b: list[list[list[int]]] = [[[1]]]
    assert a == b


test()
