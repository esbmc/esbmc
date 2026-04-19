import re

assert re.search("123", "abc123") is not None
assert re.fullmatch("abc", "abc") is not None
