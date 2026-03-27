def test_hex_with_prefix(x: str) -> bool:
    # int("0xFF", 16) should accept the prefix
    return int(x, 16) == 255

def test_binary_with_prefix(x: str) -> bool:
    return int(x, 2) == 5

def test_octal_with_prefix(x: str) -> bool:
    return int(x, 8) == 64

assert test_hex_with_prefix("0xFF")
assert test_binary_with_prefix("0b101")
assert test_octal_with_prefix("0o100")
