# repr() of a runtime str keeps its quotes (#7559).
def f(s: str) -> None:
    assert repr(s) == "ab"


f("ab")
