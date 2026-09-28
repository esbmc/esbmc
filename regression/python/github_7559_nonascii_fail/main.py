# repr() of a non-ASCII str is left unconstrained, never proved wrong (#7559).
def f(s: str) -> None:
    assert repr(s) != "'\u00e9'"


f("\u00e9")
