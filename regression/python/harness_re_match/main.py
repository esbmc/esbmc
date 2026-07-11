# Verification harness for re.match on literal patterns
# (src/python-frontend/models/re.py).
#
# re.match(pattern, string) reports whether the pattern matches at the START of
# the string. This harness uses literal patterns, which the model handles
# exactly. (Character-class patterns such as [a-z] are not soundly modelled —
# they return an unconstrained result — so they are avoided here.)
#
# This test is concrete and also runs under CPython.
#
# ENSURES:
#   E1: a literal matches an equal string
#   E2: a literal matches a string it is a prefix of (match anchors at start)
#   E3: a differing first character does not match
#   E4: a string shorter than the pattern does not match
import re

assert re.match("abc", "abc")  # E1
assert re.match("abc", "abcdef")  # E2
assert not re.match("abc", "xbc")  # E3
assert not re.match("abc", "ab")  # E4
