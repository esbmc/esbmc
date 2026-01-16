lower = "a"
upper = lower.upper()
assert upper == "A"
assert upper.lower() == "a"
# Case checks
assert "a".islower()
assert "A".isupper()
assert not "a".isupper()
assert not "A".islower()
