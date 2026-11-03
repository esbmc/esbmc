# re flag constants: IGNORECASE matches case-insensitively, ASCII and UNICODE
# change nothing for ASCII text (#8223). Results are tested by truthiness.
import re

assert re.search("AB", "xabz", re.I)
assert re.match("ab", "ABc", re.IGNORECASE)
assert re.fullmatch("[a-c]+", "ABC", re.I)
assert re.search("AB", "xabz", re.IGNORECASE | re.ASCII)
assert not re.search("AB", "xabz", re.I)
