# Negative: zfill must produce "00042", not "99999". With the inverted guard
# the receiver became a nondet string and this assertion was satisfiable,
# masking the bug. After the fix the value is concrete and the assertion fails.
assert "42".zfill(5) == "99999"
