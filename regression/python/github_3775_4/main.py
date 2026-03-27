def greet() -> str:
    return "Hi"


def test():
    x: int = greet()
    assert x == "Hi"


test()
