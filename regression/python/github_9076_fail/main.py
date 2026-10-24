# Issue #9076: a `-> bytes` function returning a bytes literal.
def encode() -> bytes:
    return b"ab"


x = encode()[2]
