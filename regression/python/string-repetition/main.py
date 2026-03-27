def test_string_repetition():
    # Basic string repetition
    s1: str = "a" * 3
    assert s1 == "aaa"

    # Reverse order (integer * string)
    s2: str = 3 * "b"
    assert s2 == "bbb"

    # Boolean multipliers
    s3: str = "x" * True
    assert s3 == "x"

    s4: str = "y" * False
    assert s4 == ""

    # Variable repetition count
    count: int = 5
    s5: str = "z" * count
    assert len(s5) == 5

    # Using isinstance to verify string type
    assert isinstance(s1, str), "s1 must be a string"
    assert isinstance(s5, str), "s5 must be a string"


test_string_repetition()
