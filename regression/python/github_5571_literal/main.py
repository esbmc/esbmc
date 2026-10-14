pairs = [("A", "B"), ("C", "D")]
s = ""
for u, v in pairs:
    s = v
assert s == "D"
