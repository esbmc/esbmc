# Exact repro from issue #4621: str() on an int variable.
x = 5
s = str(x)
assert s == "5"
