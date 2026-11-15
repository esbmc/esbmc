# len(chr(200)) is one code point, not the two UTF-8 bytes it is stored as
# (#7552).
c = chr(200)
assert len(c) == 2
