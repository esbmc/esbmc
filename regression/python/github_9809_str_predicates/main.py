def check(s: str, decimal: bool, title: bool, printable: bool) -> None:
    assert s.isdecimal() == decimal
    assert s.istitle() == title
    assert s.isprintable() == printable
    d = s.isdecimal() or s.isspace()
    assert d == decimal


check("0123", True, False, True)
check("", False, False, True)
check("Hello World", False, True, True)
check("A1 B", False, True, True)
check("Hello world", False, False, True)
check("HEllo", False, False, True)
check("a\tb", False, False, False)
check("X\n", False, True, False)
check("~\x7f", False, False, False)
