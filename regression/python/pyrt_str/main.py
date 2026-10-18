s = "abc"
assert len(s) == 3
assert s[0] == "a"
assert s[-1] == "c"
t = s + "de"
assert len(t) == 5
assert t == "abcde"
assert t != s
assert s == "abc"
assert not ""
assert "x"
assert isinstance(s, str)
assert type(s) is str
