x = 3
assert x.bit_length() == 2
s = "ab"
assert s.upper() == "AB"
c = s[0]
assert c.upper() == "A"
f = 2.5
assert f.is_integer() is False
try:
    x.upper()
    assert False
except AttributeError:
    pass
