class MyClass:
    def func3(self):
        return "Hello from func3"

    def func4(self):
        return nondet_int()

    def func2(self, a):
        return a()

    def func1(self, a, b):
        return a(b)


a = MyClass()
use3 = nondet_bool()
d = a.func1(a.func2, a.func3 if use3 else a.func4)

if use3:
    assert d == "Hello from func3"
else:
    assert isinstance(d, int)
    assert d >= 0
