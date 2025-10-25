s: str = "foo"
l: list[str] = ['a', 'b', 'c']
for ss in l:
    assert ss in s
