# A byteorder that is not a constant string is refused rather than folded.
def pick(flag: bool) -> str:
    return "big" if flag else "little"


a = bytes([1, 0])
assert int.from_bytes(a, pick(True)) == 256
