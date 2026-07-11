# Falsification counterpart for the bare single-character class fix
# (src/python-frontend/models/re.py).
#
# 'H' is uppercase, so it is not in [a-z]; asserting a match must fail. Before
# the fix this assertion could spuriously hold because the model returned a
# non-deterministic result for the unrecognised "[a-z]" pattern.
import re

assert re.match("[a-z]", "H")       # falsifiable — 'H' is not in [a-z]
