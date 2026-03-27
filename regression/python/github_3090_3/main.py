def foo(x: int) -> bool:
    assert not chr(x) == 2
    return True


assert foo(2)
