def foo(x: str) -> None:
    # count=1 replaces only the first match, not every match.
    assert x.replace("aa", "x", 1) == "x-bb-x"


foo("aa-bb-aa")
