s = "abc"
assert s[0] == "a"
assert s[-1] == "c"

try:
    _ = s[0]
    assert False, "IndexError expected"
except IndexError:
    pass
