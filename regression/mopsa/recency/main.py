class C:
    def __init__(self, x):
        self.x = x

def alloc(x):
    return C(x)

a = alloc(1)

b = alloc(2)

a.y = 20

a.y = a.y + 1

c = alloc(3)

res = a.x
