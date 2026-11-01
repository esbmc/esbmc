# The negative chars of a UTF-8 literal used to be read back as their
# magnitude: 'é' (C3 A9) became "=W", the E0 lead byte a space and the
# BF continuation byte 'A'.
d = "é"
assert not d.isupper()
assert "ࠀx".lstrip() == "ࠀx"
assert "ࠀx".strip() == "ࠀx"
assert "ࠀx".lstrip(" ") == "ࠀx"
assert "xÿ".rstrip("A") == "xÿ"
