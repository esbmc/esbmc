a = nondet_str()
b = nondet_str()
c = nondet_str()
l = [a, b, c]
assert l[0] == a
assert l[1] == b
assert l[2] == c
