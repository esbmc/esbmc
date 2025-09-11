def add_suffix(x: str) -> str:
    return x + "_suffix"


# This should use strncmp since it's not an identity function
assert add_suffix("test") == "test_suffix"
