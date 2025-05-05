assert oct(8) == "0o10"
assert oct(-9) == "-0o11"
assert oct(0) == "0o0"
assert oct(-0) == "0o0"  # Python treats -0 as 0 for oct()
assert oct(1) == "0o1"
assert oct(7) == "0o7"
assert oct(8) == "0o10"
assert oct(64) == "0o100"
assert oct(-9) == "-0o11"
assert oct(-64) == "-0o100"
assert oct(123456789) == "0o726746425"
assert oct(-123456789) == "-0o726746425"

