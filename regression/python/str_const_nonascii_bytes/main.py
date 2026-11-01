# The negative chars of a UTF-8 literal used to be read back as their
# magnitude: 'é' (C3 A9) became "=W", and the E0 lead byte a space.
d = "é"
assert not d.isupper()
s = "ࠀx"
assert s.lstrip() == s
assert s.strip() == s
