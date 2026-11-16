# The negative chars of a UTF-8 literal used to be read back as their
# magnitude: the E0 lead byte became a space and the BF continuation
# byte 'A'.
assert "ࠀx".lstrip() == "ࠀx"
assert "ࠀx".strip() == "ࠀx"
assert "ࠀx".lstrip(" ") == "ࠀx"
assert "xÿ".rstrip("A") == "xÿ"
