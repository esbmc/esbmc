def foo(x: str) -> None:
    s = x.replace("aa", "x", 5)
    assert s == "x-bb-x"


foo("aa-bb-aa")
