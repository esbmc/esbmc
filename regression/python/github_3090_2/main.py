i1: int = 102
i2: int = 111
i3: int = 111
s: str = ""
s += chr(i1)
s += chr(i2)
s += chr(i3)
assert s == "foo"
