# Verification harness for the string module constants
# (src/python-frontend/models/string.py).
#
# The model defines the standard character-class constants. This harness pins
# the short-string lengths and the boundary characters of each range via O(1)
# indexing. It is intrinsic-free, so it also runs under CPython.
#
# The program is fully concrete, so it is verified with plain BMC. Lengths are
# checked only on the short constants (digits, octdigits): len() scans to the
# terminator, and scanning the 52-char ascii_letters would demand a large
# unwind bound that stresses the solver. ascii_letters' layout is pinned
# structurally by its boundary indices instead, which need no scan.
#
# ENSURES:
#   E1: digits and octdigits have their documented lengths
#   E2: the digit / lowercase / uppercase ranges start and end at the right chars
#   E3: ascii_letters is ascii_lowercase followed by ascii_uppercase
import string

# E1 — short-string lengths
assert len(string.digits) == 10
assert len(string.octdigits) == 8

# E2 — range boundaries via O(1) indexing
assert string.digits[0] == '0'
assert string.digits[9] == '9'
assert string.ascii_lowercase[0] == 'a'
assert string.ascii_lowercase[25] == 'z'
assert string.ascii_uppercase[0] == 'A'
assert string.ascii_uppercase[25] == 'Z'

# E3 — ascii_letters == ascii_lowercase then ascii_uppercase
assert string.ascii_letters[0] == 'a'
assert string.ascii_letters[25] == 'z'
assert string.ascii_letters[26] == 'A'
assert string.ascii_letters[51] == 'Z'

assert string.hexdigits[0] == '0'
