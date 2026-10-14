def to_hex(x: int) -> str:
    return hex(x)


def to_oct(x: int) -> str:
    return oct(x)


# Companion runtime OMs for hex/oct on symbolic int inputs.
assert to_hex(0) == "0x0"
assert to_hex(255) == "0xff"
assert to_hex(-16) == "-0x10"
assert to_oct(0) == "0o0"
assert to_oct(8) == "0o10"
assert to_oct(-9) == "-0o11"
