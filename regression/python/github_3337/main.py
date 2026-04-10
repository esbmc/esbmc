b = True
s: str = "" if b else "foo"
assert len(s) == 0
