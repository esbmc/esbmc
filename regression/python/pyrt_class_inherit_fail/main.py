class Base:
    def __init__(self, x):
        self.x = x

    def describe(self):
        return self.x

    def __add__(self, other):
        return self.describe() + other.describe()


class Child(Base):
    def describe(self):
        return self.x * 10


assert Child(1) + Child(2) == 3
