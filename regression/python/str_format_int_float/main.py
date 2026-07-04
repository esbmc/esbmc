# An integer value with a float presentation type (f/F/e/E/g/G) formats as a
# float, matching CPython. Previously "{:.2f}".format(5) fell to a nondet
# fallback (the int branch only handled d/x/X/o/b).
assert "{:.2f}".format(5) == "5.00"
assert "{:f}".format(5) == "5.000000"
assert "{:.1e}".format(1234) == "1.2e+03"
assert "{:.1E}".format(1234) == "1.2E+03"
assert "{:g}".format(100) == "100"
assert "{:8.2f}".format(5) == "    5.00"
# Booleans (which are ints) fold too.
assert "{:.2f}".format(True) == "1.00"
# CPython converts the int to a float first, so ESBMC's (double) cast matches
# its rounding even past 2^53 (9007199254740993 -> ...992, not the exact int).
assert "{:f}".format(9007199254740993) == "9007199254740992.000000"
