def main() -> None:
    # Two-string maketrans: map each x[i] to y[i].
    s = "hello".translate(str.maketrans("el", "ip"))
    assert s == "hippo"
    assert len(s) == 5 and s[1] == "i"

    # Three-string maketrans: the third argument's characters are deleted.
    d = "hello world".translate(str.maketrans("", "", "lo"))
    assert d == "he wrd"

    # A variable receiver works, and characters absent from the table are kept.
    x = "abcabc"
    assert x.translate(str.maketrans("abc", "xyz")) == "xyzxyz"
    assert "keep".translate(str.maketrans("Q", "Z")) == "keep"


main()
