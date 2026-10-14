def main() -> None:
    # CPython puts the extra fill char on the LEFT when both the margin
    # and the width are odd: left = marg//2 + (marg & width & 1).
    assert "ab".center(7, "-") == "---ab--"
    assert "ab".center(5) == "  ab "
    assert "a".center(4) == " a  "
    assert "abc".center(6, "*") == "*abc**"
    # Even split and degenerate widths unchanged.
    assert "ab".center(6) == "  ab  "
    assert "a".center(5) == "  a  "
    assert "ab".center(2) == "ab"
    assert "ab".center(1) == "ab"
    assert "".center(3, "x") == "xxx"


main()
