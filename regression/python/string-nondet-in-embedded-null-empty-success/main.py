s = nondet_string(4)
assume(s == "a\0b")

assert "" in s
assert "a\0b" in s
