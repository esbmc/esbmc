def append_one(a):
    return a + [1]


y = append_one([2])
assert y != [2, 1]
