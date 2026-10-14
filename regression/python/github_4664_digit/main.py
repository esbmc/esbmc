import re

# CPython: \d+ matches the leading "123" of "123abc" -> truthy Match.
m = re.match(r"\d+", "123abc")
assert m
