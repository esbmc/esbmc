def is_digit(c: str) -> bool:
    return c >= '0' and c <= '9'


def foo(x: str) -> None:
    assert len(x) > 2
    assert is_digit(x[0])
    assert is_digit(x[1])
    assert is_digit(x[2])


x: str = "123"
foo(x)
