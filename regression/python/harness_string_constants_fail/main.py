# Falsification harness for the string module constants
# (src/python-frontend/models/string.py).
#
# string.digits starts with '0', so a wrong boundary-character claim must be
# falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: string.digits[0] == '1'.  The first digit is '0'.
import string

assert string.digits[0] == '1'      # F1 — falsifiable (digits starts with '0')
