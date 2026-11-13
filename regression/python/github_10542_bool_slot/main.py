# #10542: a bool stored into a list of ints reads back as a bool.
l = [1, 2]
l[0] = True
assert type(l[0]) is bool
assert l[0] == 1
assert l[1] == 2
