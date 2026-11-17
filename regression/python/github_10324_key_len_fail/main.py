words = ["bb", "a", "ccc", "dd", "e"]
s = sorted(words, key=len)
assert s == ["a", "e", "bb", "dd", "ccc"]
r = sorted(words, key=len, reverse=True)
assert r == ["ccc", "dd", "bb", "e", "a"]
