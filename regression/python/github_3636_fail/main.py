def test_isspace_fail() -> None:
    # "hello" is not all whitespace, so isspace() returns False
    # asserting True should fail verification
    assert "hello".isspace()


test_isspace_fail()
