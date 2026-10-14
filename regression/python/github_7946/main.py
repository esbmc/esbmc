# byteorder defaults to "big" since Python 3.11 (#7946).
a = bytes([1, 0])
assert int.from_bytes(a) == 256
assert int.from_bytes(a, signed=True) == 256
