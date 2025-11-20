def test_string_concatenation():
    assert "a" + "b" == "ab"
    assert "hello" + "" == "hello"
    assert "" + "world" == "world"

    x = "foo"
    y = "bar"
    assert x + y == "foobar"

    assert "answer: " + str(42) == "answer: 42"
    assert "α" + "β" == "αβ"

test_string_concatenation()
