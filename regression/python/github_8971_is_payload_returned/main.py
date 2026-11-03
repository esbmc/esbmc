class MyClass:
    def __init__(self, v):
        self.v = v

    def func1(self):
        return self.v

    def func2(self):
        a = self
        return a.func1()


if nondet_bool():
    v = "Hello from func1 in MyClass"
else:
    v = nondet_int()
a = MyClass(v)
b = a.func2()
assert b is v
