# A float constant folded through %.17g printed 10000000000000002.0 in fixed
# notation, so this CPython-false assertion verified (#7559).
assert str(10000000000000002.0) == "10000000000000002"
