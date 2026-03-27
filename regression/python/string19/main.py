def identity_string(x: str) -> str:
    return x


single_char = "a"
assert identity_string(single_char) == single_char
assert single_char == identity_string(single_char)
