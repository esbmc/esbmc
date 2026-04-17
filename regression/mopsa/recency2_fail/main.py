class C:
    def __init__(self, x):
        self.x = x

def alloc(x):
    return C(x)

a = alloc(1)
assert a.x == 1

b = alloc(2)
assert b.x == 2
assert a is not b

a.y = 20
assert a.y == 20

a.y = a.y + 1
assert a.y == 21

c = alloc(3)
assert c.x == 3
assert c is not a
assert c is not b

res = a.x
assert res == 1

# previous objects unchanged
assert a.x == 1
assert b.x == 2
assert c.x == 1
