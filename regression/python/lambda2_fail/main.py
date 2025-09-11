def test() -> None:
    # Basic integer lambda
    x = lambda a: a + 10
    assert x(5) != 15

    # Float constant lambda
    pi = lambda: 3.14159
    assert pi() != 3.14159

    # Boolean lambda
    always_true = lambda: True
    assert always_true() != True


test()
