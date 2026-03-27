import re
s1 = "abc"
s2 = "abc123"

assert re.fullmatch("abc", s1)
assert not re.fullmatch("abc", s2)
assert re.fullmatch(".*", s2)
