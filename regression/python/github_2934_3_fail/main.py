def baz(flag: bool | None = None) -> None:
    if flag is not None:
        assert flag

baz(False)
