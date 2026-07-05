a = [1, 2, 3, 4]
a[1] = 9            # plain index store
assert a[1] == 9

m = [[1, 2], [3, 4]]
m[0][1] = 7         # nested index store
assert m[0][1] == 7

b = a[1:3]          # slice read (rvalue) is fine
assert b[0] == 9 and b[1] == 3
