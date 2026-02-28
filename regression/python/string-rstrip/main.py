def main() -> None:
    assert "  hello  ".rstrip() == "  hello"
    assert "world\t\n".rstrip() == "world"
    assert "nochange".rstrip() == "nochange"
    assert "trailingspace ".rstrip() == "trailingspace"
    assert "tabs\t\t".rstrip() == "tabs"
    assert "\n\r\f\v".rstrip() == ""
    assert "".rstrip() == ""
    assert "x".rstrip() == "x"
    assert "x \t\n".rstrip() == "x"
    assert "  lead and trail  ".rstrip() == "  lead and trail"
    assert "middle  space".rstrip() == "middle  space"
    assert "  onlyleading".rstrip() == "  onlyleading"
    assert "mix\t \nend".rstrip() == "mix\t \nend"
    assert "   ".rstrip() == ""


main()
