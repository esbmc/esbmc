# repr() of runtime values: numbers and bools as str() renders them, an ASCII
# str quoted and escaped as CPython does (#7559).
def f(s: str, t: str, n: int, b: bool) -> None:
    assert f"{s!r}" == "'a\\'b\"c'"
    assert repr(t) == "'\\x01\\x7f\\r'"
    assert ascii(t) == "'\\x01\\x7f\\r'"
    assert f"{n!r}" == "42" and f"{b!a}" == "False"
    try:
        ascii(s)
    except TypeError:
        assert False


f("a'b\"c", "\x01\x7f\r", 42, False)
