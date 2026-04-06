# Three-target chained assignment with list comprehension
a = b = c = [i * i for i in range(3)]
assert a == [0, 1, 4]
assert b == [0, 1, 4]
assert c == [0, 1, 4]
assert a[0] == 0
assert b[1] == 1
assert c[2] == 4
