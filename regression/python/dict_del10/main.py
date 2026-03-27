d = {"k": 1}
key = "k"
del d[key]     # Should delete "k"
assert "k" not in d
