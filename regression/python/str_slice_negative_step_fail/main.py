# "abcdef"[5:0:-1] is "fedcb" (len 5), not "fedcba" — the old wrong result.
assert "abcdef"[5:0:-1] == "fedcba"
