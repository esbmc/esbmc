# Issue #5114: str.partition(sep) must return the (before, sep, after) 3-tuple,
# not a mistyped 1-element scalar. Verify the tuple shape, contents (separator
# found and not found), len(), indexing and unpacking all match CPython.
def main() -> None:
    t = "a.b.c".partition(".")
    assert len(t) == 3
    assert t[0] == "a"
    assert t[1] == "."
    assert t[2] == "b.c"

    # Separator not found: (whole, "", "")
    u = "a-b-c".partition(".")
    assert len(u) == 3
    assert u[0] == "a-b-c"
    assert u[1] == ""
    assert u[2] == ""

    # Tuple unpacking
    before, sep, after = "key=value".partition("=")
    assert before == "key"
    assert sep == "="
    assert after == "value"


main()
