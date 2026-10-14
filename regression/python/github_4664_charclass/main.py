import re

# CPython: [a-z]+ matches the leading "abc" of "abc123" -> truthy Match.
m = re.match(r"[a-z]+", "abc123")
assert m

# Non-match: "123" has no leading lowercase.
m2 = re.match(r"[a-z]+", "123")
assert not m2
