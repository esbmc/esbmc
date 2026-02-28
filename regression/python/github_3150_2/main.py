s = "foo"
for c in s:
    i: int = ord(c)
    assert len(c) == 1
    assert isinstance(i, int)
    assert i == ord(c)  # trivially true
