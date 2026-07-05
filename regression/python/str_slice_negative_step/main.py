# A string slice with a negative step and explicit bounds now folds
# correctly. The runtime string-slice path mishandled a negative step (wrong
# content and an unconstrained length: both len==5 and len==6 verified);
# consteval's slice computation matches CPython, so a constant string slice
# is folded at conversion time. Covers literal, variable (module/local), and
# concatenated const receivers, and confirms the length is now constrained.
assert "abcdef"[5:0:-1] == "fedcb"
assert "abcdef"[4:1:-1] == "edc"
assert len("abcdef"[5:0:-1]) == 5
s = "abcdef"
assert s[5:0:-1] == "fedcb"
assert ("ab" + "cdef")[5:0:-1] == "fedcb"
# positive step and full reverse remain correct
assert "abcdef"[1:5:2] == "bd"
assert "abcdef"[::-1] == "fedcba"
# The fold gates on the converted str type, so a bytes literal slice is NOT
# folded to str (it stays bytes, handled by its own runtime path).
assert isinstance(b"abcdef"[1:4], bytes)
