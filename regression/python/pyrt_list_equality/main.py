# Two lists holding equal items are equal, whether or not they are the same
# object -- element-wise, as tuples and strings already compared.
assert [1, 2] == [1, 2]
assert [1, 2] != [1, 3]
assert [1] != [1, 2]
assert [] == []
a = [1, 2]
b = a
assert a == b
