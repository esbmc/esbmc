def is_str(x: str) -> bool:
    return isinstance(x, str)


assert not is_str("a")
