# str.format() now supports the {:c} (integer -> character) and {:%}
# (float -> percentage) presentation types, which previously fell to a nondet
# fallback.
assert "{:c}".format(65) == "A"
assert "{:c}".format(97) == "a"
assert "{:5c}".format(65) == "    A"
assert "{:<5c}".format(65) == "A    "
assert "{:%}".format(0.25) == "25.000000%"
assert "{:.1%}".format(0.5) == "50.0%"
assert "{:.0%}".format(0.5) == "50%"
assert "{:8.2%}".format(0.5) == "  50.00%"
assert "{:+.1%}".format(0.5) == "+50.0%"
