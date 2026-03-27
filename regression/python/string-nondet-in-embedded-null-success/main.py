s = nondet_string(4)
assume(s == "a\0b")

assert "\0b" in s
assert "a\0" in s
assert "c\0" not in s
