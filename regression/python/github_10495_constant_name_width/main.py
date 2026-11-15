# A name bound once to an integer constant must retain literal behavior.
W = 8
assert "ab".ljust(W) == "ab      "
assert "ab".rjust(W) == "      ab"
assert "ab".center(W) == "   ab   "
assert "ab".zfill(W) == "000000ab"

# A loop width no larger than the string needs no padding.
for i in range(3):
    assert "ab".rjust(i) == "ab"
