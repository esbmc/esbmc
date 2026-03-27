def test_binary(x: str) -> bool:
    return int(x, 2) == 5


def test_hex(x: str) -> bool:
    return int(x, 16) == 255


def test_octal(x: str) -> bool:
    return int(x, 8) == 64


assert test_binary("101")
assert test_hex("FF")
assert test_octal("100")
