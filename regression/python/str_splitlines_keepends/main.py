# str.splitlines(keepends=True) retains the line terminators, which was
# previously unsupported. Covers \n, \r\n, and the keyword form; keepends=False
# is unchanged.
s = "a\nb"
assert s.splitlines(True) == ["a\n", "b"]
assert s.splitlines(keepends=True) == ["a\n", "b"]
assert "a\r\nb".splitlines(True) == ["a\r\n", "b"]
assert s.splitlines() == ["a", "b"]
