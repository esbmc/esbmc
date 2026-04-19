# Annotation says float but function returns int - Python allows this
def get_value() -> int:
    return 42


def test() -> None:
    x: float = get_value()
    assert x == 42


test()
