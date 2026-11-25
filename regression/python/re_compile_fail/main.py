import re

p = re.compile(r"b")
assert p.search("acd")
