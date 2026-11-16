class A:
    def f1(self):
        return 1


a = A()
if True:
    c, d = a.f1, 0
x = c()
assert x == 2
