class A:

    def __init__(self):
        self.b = "Hello"

    def func(self):
        return self.b


a = A()
override = nondet_bool()
if override:
    a.b = 5
r = a.func()
if not override:
    r + 1
