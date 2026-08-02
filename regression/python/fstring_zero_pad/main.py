x = 7
# Zero-padding shorthand and width with integer presentation type.
assert f"{x:03d}" == "007"
assert f"{x:3d}" == "  7"
assert f"{x:03}" == "007"
assert f"{x:<3d}" == "7  "
assert f"{x:d}" == "7"
# Sign-aware '=' alignment: padding between the sign and the digits.
n = -42
assert f"{n:05d}" == "-0042"
assert f"{n:05}" == "-0042"
# Width narrower than the value: no padding.
big = 12345
assert f"{big:03d}" == "12345"
# String alignment on a variable receiver; zero-pad on a string is
# left-aligned (CPython: format('ab', '05') == 'ab000').
s = "ab"
assert f"{s:>5}" == "   ab"
assert f"{s:5}" == "ab   "
assert f"{s:s}" == "ab"
assert f"{s:05}" == "ab000"
# Float precision composed with fill/align/width.
v = 1.5
assert f"{v:07.2f}" == "0001.50"
assert f"{v:8.2f}" == "    1.50"
assert f"{v:<8.2f}" == "1.50    "
w = -1.5
assert f"{w:07.2f}" == "-001.50"
