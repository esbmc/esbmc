# A byteorder that is not a string literal used to fold to little-endian
# (#7542); it is refused instead.
def pick(flag: bool) -> str:
    return "big" if flag else "little"


a = bytes([1, 0])
assert int.from_bytes(a, pick(True)) == 256
