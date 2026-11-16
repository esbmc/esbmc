d = [["a", "b"], ["b"]]
e = {k for j in d for k in j}
assert len(e) == 2
assert "a" in e
