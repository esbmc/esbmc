def foo(x: int) -> bool:
    assert chr(x) == 2
    return True


assert foo(2)
