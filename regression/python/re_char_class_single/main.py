# Regression test for the bare single-character class [x-y] in re.match /
# re.search / re.fullmatch (src/python-frontend/models/re.py).
#
# Before the fix, a pattern like "[a-z]" (no quantifier) was not recognised by
# try_match_char_class_range (which required a length-6 quantified pattern), so
# match() fell through to its non-deterministic fallback and returned an
# unconstrained bool. This test pins the correct behaviour: "[x-y]" matches
# exactly one character in the range.
import re

# match: anchors at the start, one in-range char suffices.
assert re.match("[a-z]", "h")
assert re.match("[a-z]", "hello")
assert not re.match("[a-z]", "H")
assert not re.match("[a-z]", "")
assert re.match("[0-9]", "5")
assert not re.match("[0-9]", "x")

# fullmatch: the whole string must be exactly one in-range char.
assert re.fullmatch("[a-z]", "h")
assert not re.fullmatch("[a-z]", "hh")
assert not re.fullmatch("[a-z]", "H")

# search: an in-range char anywhere in the string.
assert re.search("[0-9]", "abc5")
assert not re.search("[0-9]", "abc")
