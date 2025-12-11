# Test nondet_str() with assignment and conditional modification

s = nondet_str()

# If s is empty, change it to a non-empty string
if s == "":
    s = "x"

# After the conditional, s should never be empty
assert s != ""

