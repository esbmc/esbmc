import re

s = "abc"
assert not re.match("[a-z]+", s)
