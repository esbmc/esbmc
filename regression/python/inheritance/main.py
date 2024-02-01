class Base:
    def __init__(self, x:int):
        self.x:int = x

class Derived(Base):
    pass

a = 10
obj = Derived(a)
assert obj.x == 10
