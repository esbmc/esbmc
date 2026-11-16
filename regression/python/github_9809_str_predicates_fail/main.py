def check(s: str) -> None:
    assert s.istitle() and s.isprintable() and not s.isdecimal()


check("Abcd")
check("Ab\tcd")
