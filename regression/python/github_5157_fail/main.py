# Negative companion to github_5157: the same value-carrying comparison must
# still detect a genuine mismatch (bin(3) is "0b11", not "0b10"). Guards
# against the fix degenerating into an always-true string comparison.
b = bin(3)
assert b == "0b10"
