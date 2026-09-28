# Verification harness for re.search and re.fullmatch on literal patterns
# (src/python-frontend/models/re.py).
#
# search finds the pattern anywhere in the string; fullmatch requires the
# pattern to match the entire string. Literal patterns are used (see
# harness_re_match for why character classes are avoided).
#
# This test is concrete and also runs under CPython.
#
# ENSURES:
#   E1: fullmatch succeeds only when the whole string equals the pattern
#   E2: fullmatch fails when the string has trailing characters
#   E3: search finds a substring anywhere in the string
#   E4: search fails when the substring is absent
import re

assert re.fullmatch("abc", "abc")  # E1
assert not re.fullmatch("abc", "abcd")  # E2
assert re.search("bc", "abcd")  # E3
assert not re.search("xy", "abcd")  # E4
