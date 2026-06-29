def main() -> None:
    # str.rindex is str.rfind that raises ValueError when the substring is not
    # found (the right-side analogue of str.index). It was unmodelled before.
    assert "abcabc".rindex("b") == 4
    assert "abcabc".rindex("c") == 5
    assert "banana".rindex("a") == 5
    assert "hello".rindex("h") == 0

    # Optional start/end search window, like rfind.
    assert "abcabc".rindex("b", 0, 3) == 1

    # The result is an int and composes in arithmetic.
    assert "abcabc".rindex("c") + 1 == 6

    # A variable receiver works too.
    s = "abcabc"
    assert s.rindex("a") == 3

    # Not found raises a catchable ValueError (unlike rfind, which returns -1).
    try:
        "abc".rindex("z")
        assert False
    except ValueError:
        pass


main()
