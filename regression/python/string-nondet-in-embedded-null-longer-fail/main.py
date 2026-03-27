s = nondet_string(4)
assume(s == "a\0b")

assert "a\0bc" in s
