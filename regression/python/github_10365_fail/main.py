class A:
    def __init__(self, flag):
        if flag:
            self.smth = self.func2
        else:
            self.smth = self.func3

    def func(self):
        return self.smth()

    def func2(self):
        return "Hello"

    def func3(self):
        return 42


flag = nondet_bool()
a = A(flag)
b = a.func()
if flag:
    assert isinstance(b, str)
    assert b == "Hello"
else:
    assert isinstance(b, int)
    assert b == 43
c = a.smth()
assert c == b
assert a.func2() == "Hello"
