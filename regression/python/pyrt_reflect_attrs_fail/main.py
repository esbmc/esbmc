class A:
    def __init__(self):
        self.x = 1


a = A()
print(getattr(a, "z"))
