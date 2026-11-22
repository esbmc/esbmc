class A:
    def __init__(self):
        self.b = "Hello"

    def func(self):
        return self.b


a = A()
flag = nondet_bool()
if flag:
    a.b = 5
r = a.func()
if flag:
    assert r == 5
else:
    assert r == "Hellp"
