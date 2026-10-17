def is_a(v, hint) -> bool:
    return isinstance(v, hint)


assert is_a(ValueError("x"), Exception)
assert is_a(1, Exception)
