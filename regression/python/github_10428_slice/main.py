# A float copied by a slice reads back as itself, not as the first float
# stored in any list (#10428).
a = [1.0, 2.0]
b = a[:]
assert b[1] == 2.0
assert b[0] == 1.0
