# Issue #9076: a `-> bytes` function returning a bytes literal.
def encode() -> bytes:
    return b"\xff\x00z"


b = encode()
assert len(b) == 3
assert b[0] == 255
assert b[1] == 0
assert b[2] == 122
