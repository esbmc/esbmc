def main() -> None:
    # str.isascii(): every char < 128; True on the empty string.
    assert "abc".isascii()
    assert "".isascii()
    assert "a1 ~\t".isascii()
    assert not "\x80".isascii()

    # str.isdecimal(): every char a decimal digit; False on the empty string.
    assert "123".isdecimal()
    assert not "".isdecimal()
    assert not "12a".isdecimal()
    assert not "1 2".isdecimal()

    # str.isprintable(): every char printable (space is, tab/DEL are not);
    # True on the empty string.
    assert "abc 123 ~".isprintable()
    assert "".isprintable()
    assert not "a\tb".isprintable()
    assert not "x\x7f".isprintable()


main()
