# Issue #9076: a bytes literal return must not be folded as a str.
def encode():
    return b"ab"


b = encode()
assert b[0] == 97
assert b[1] == 98
