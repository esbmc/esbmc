class A:
    def __init__(self, flag):
        if flag:
            self.smth = self.func2
        else:
            self.smth = self.func3

    def func(self):
        return self.smth()

    def func2(self):
        return 1

    def func3(self):
        return 42


flag = nondet_bool()
a = A(flag)
b = a.func()
if flag:
    assert b == 1
else:
    assert b == 42
