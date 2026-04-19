def empty_identity(x: str, y: int) -> str:
    if y:
        assert False
    else:
        return x

assert empty_identity("", 1) == ""
