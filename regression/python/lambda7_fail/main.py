def test() -> None:
    # int lambda, but wrong expected value
    f_int:int = lambda a: a + 10
    assert f_int(5) == 20  # should fail

    # bool lambda, wrong expectation
    f_bool:bool = lambda a: a and True
    assert f_bool(False) is True  # should fail

test()

