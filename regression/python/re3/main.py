import re

s = "hello world"
assert re.search(".*", s)
assert not re.search("foo", s)
