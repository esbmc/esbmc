
s = nondet_string(3)
assume(s == "abc")
c1 = s[0]
assert c1 == "a"
c2 = s[1]
assert c2 == "b"
c3 = s[2]
assert c3 == "c"
