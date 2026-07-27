# An empty "{}" replacement field renders a float with CPython's str(), which
# uses the fewest significant digits that read back as the same double. ESBMC
# printed 6 significant digits, so "{}".format(1.23456789) folded to "1.23457".
assert "{}".format(1.23456789) == "1.23456789"
assert "{}".format(3.14159265358979) == "3.14159265358979"
assert "{}".format(123456789.5) == "123456789.5"
assert "{}".format(0.30000000000000004) == "0.30000000000000004"
assert "{}".format(-1.23456789) == "-1.23456789"

# Short values were already correct and must stay so.
assert "{}".format(0.1) == "0.1"
assert "{}".format(2.5) == "2.5"

# Whole-number floats keep the ".0" the dedicated branch adds.
assert "{}".format(1.0) == "1.0"
assert "{}".format(-1.0) == "-1.0"
assert "{}".format(-0.0) == "-0.0"
assert "{}".format(1000000.0) == "1000000.0"

# Fixed/exponential cut-over matches CPython on both sides.
assert "{}".format(0.0001) == "0.0001"
assert "{}".format(1e-05) == "1e-05"
assert "{}".format(1e15) == "1000000000000000.0"
assert "{}".format(1e16) == "1e+16"
assert "{}".format(1.5e20) == "1.5e+20"
assert "{}".format(1e300) == "1e+300"

# A field mixed with an int argument.
assert "x={} y={}".format(1.23456789, -4) == "x=1.23456789 y=-4"
