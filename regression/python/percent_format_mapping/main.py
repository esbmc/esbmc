def main() -> None:
    # The %(name)s mapping form of printf-style % formatting (right-hand dict)
    # was rejected ("unsupported conversion '%('"). It now looks each key up in
    # the dict.
    assert "%(x)s" % {"x": "hi"} == "hi"
    assert "%(n)d" % {"n": 5} == "5"
    assert "%(a)s=%(b)s" % {"a": "k", "b": "v"} == "k=v"
    assert "%(x)d+%(y)d" % {"x": 2, "y": 3} == "2+3"

    # A key may be referenced more than once; %% is still a literal percent.
    assert "%(x)s%(x)s" % {"x": "ab"} == "abab"
    assert "%(p)d%%" % {"p": 50} == "50%"

    # Positional formatting is unchanged.
    assert "%s=%d" % ("x", 5) == "x=5"


main()
