def test() -> None:
    outer:int = lambda x: (lambda y: x + y)
    inner:int = outer(5)

test()
