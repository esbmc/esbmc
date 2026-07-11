# Falsification harness for re.match on literal patterns
# (src/python-frontend/models/re.py).
#
# re.match anchors at the start, so a literal that does not begin the string
# does not match; asserting a match must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: re.match("abc", "xyz").  "abc" is not a prefix of "xyz".
import re

assert re.match("abc", "xyz")       # F1 — falsifiable (no prefix match)
