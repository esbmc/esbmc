def test() -> None:
    # int lambda
    f_int: int = lambda a: a + 10
    assert f_int(5) == 15

    # bool lambda
    f_bool: bool = lambda a: a and True
    assert f_bool(True) is True
    assert f_bool(False) is False

    # string lambda
    f_str: str = lambda s: s + " world"
    assert f_str("hello") == "hello world"


test()
