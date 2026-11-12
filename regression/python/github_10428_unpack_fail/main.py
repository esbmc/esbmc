# #10428
a = [1.0, 2.0, 3.0]
first, *rest = a
assert rest[1] == 1.0
