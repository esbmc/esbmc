a = {1, 2}
b = {1, 2, 3}
c = {4}
assert a <= b
assert a < b
assert not (b <= a)
assert not (b < a)
assert b >= a
assert b > a
assert not (a >= b)
assert not (a > b)
assert a <= a
assert a >= a
assert not (a < a)
assert not (a > a)
assert not (a <= c)
assert not (c <= a)
assert not (a < c)
assert not (a >= c)
assert not (a > c)
assert {1} >= {1}
assert not ({1} >= {3})
