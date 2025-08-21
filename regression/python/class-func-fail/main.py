def foo():
    assert False

class MyClass:
    def bar(self):
        foo()

obj = MyClass()
obj.bar()

