import re

assert re.match(".*", "")
assert re.match(".*", "abc")
assert re.search(".*", "xyz")
assert re.fullmatch(".*", "whatever")
