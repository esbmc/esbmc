# A float copied into a starred target reads back as itself (#10428).
a = [1.0, 2.0, 3.0]
first, *rest = a
assert rest[1] == 3.0
