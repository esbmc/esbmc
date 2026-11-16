class A:
    def f1(self):
        return 1


a = A()
for i in range(0):
    c, d = a.f1, 0
x = c()
assert x == 1
