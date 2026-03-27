s = nondet_string(6)
assume(s == "Python")
sub1 = s[0:3]
assert sub1 == "Pyt"
sub2 = s[3:6]
assert sub2 == "hon"
sub3 = s[:2]
assert sub3 == "Py"
