def test_auto_hex(x: str) -> bool:
    return int(x, 0) == 255

def test_auto_binary(x: str) -> bool:
    return int(x, 0) == 5

def test_auto_octal(x: str) -> bool:
    return int(x, 0) == 64

assert test_auto_hex("0xFF")
assert test_auto_binary("0b101")
assert test_auto_octal("0o100")
