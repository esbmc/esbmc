def label() -> None:
    # Annotation says int, but the value is a str. Python keeps the str.
    n: int = "hello"
    assert n == "hello"


label()
