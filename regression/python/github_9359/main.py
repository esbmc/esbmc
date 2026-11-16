# Issue #9359: an int compares with a float by exact value.
a = 9007199254740993
b = 9007199254740992.0
assert a != b
assert not (a == b)
assert a > b
assert a >= b
assert not (a < b)
assert not (a <= b)
assert b < a
assert b <= a
assert not (b > a)
assert 9007199254740992 == b
assert 0 == -0.0
assert 1.5 < a > b
assert (a, 1) > (b, 2)
assert (a, 0) != (b, 0)
assert [a] != [b]
assert [9007199254740992] == [b]
