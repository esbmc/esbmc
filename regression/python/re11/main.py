import re

assert re.match("[A-Z]+", "ABC")
assert re.match("[0-9]*", "12345")
assert not re.match("[a-z]+", "123")
