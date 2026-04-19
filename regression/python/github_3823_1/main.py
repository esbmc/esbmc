# Basic global variable access from class method
x = 1

class C:
    def f(self):
        return x

result = C().f()
assert result == 1
