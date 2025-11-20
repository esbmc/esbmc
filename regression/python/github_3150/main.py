s = "foo"
for c in s:
    i: int = ord(c)
    assert i == 102 or i == 111
