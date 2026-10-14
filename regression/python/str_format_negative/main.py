# str.format() / format() now handle a negative literal argument. Previously
# a negative (parsed as UnaryOp(USub, Constant)) was rejected, degrading the
# whole call to a nondet string, so even "{}".format(-5) could not verify.
assert "{}".format(-5) == "-5"
assert "{:d}".format(-5) == "-5"
assert "{:x}".format(-255) == "-ff"
assert "{:5d}".format(-42) == "  -42"
assert "{:.1f}".format(-2.5) == "-2.5"
assert "{} and {}".format(-1, -2) == "-1 and -2"
assert "{}-{}".format(5, -3) == "5--3"
assert format(-7) == "-7"
# -0 renders "0" (not "-0"); -True/-False fold numerically.
assert "{}".format(-0) == "0"
assert "{}".format(-True) == "-1"
assert "{}".format(-False) == "0"
