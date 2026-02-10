class A:
    pass

a = A()

class B:
    def __init__(self, x):
        self.x = x

    def __add__(self, c):
        return self.x + c.x
    
b1 = B(1)
b2 = B(2)

res1 = b1 + b2

class C:
    def __init__(self, x):
        self.x = x

    def __radd__(self, c):
        return self.x + 100

c1 = C(1)
res2 = a + c1
