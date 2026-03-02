def test() -> None:
    outer = lambda x: (lambda y: x + y)
    inner = outer(5)


test()
