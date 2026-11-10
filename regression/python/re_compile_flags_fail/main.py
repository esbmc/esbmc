import re

p = re.compile(r"B", re.IGNORECASE)
assert not p.search("abc")
