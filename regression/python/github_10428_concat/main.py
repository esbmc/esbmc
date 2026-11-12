# A float copied by + reads back as itself (#10428).
b = [1.0, 2.0] + []
assert b[1] == 2.0
c = [1.0] + [2.0]
assert [1, 2] == c
