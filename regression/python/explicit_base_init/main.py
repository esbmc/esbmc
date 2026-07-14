# A subclass calling its base class's __init__ explicitly via the class name
# (`Base.__init__(self, ...)`), the pre-super() idiom. This previously aborted
# with "_init_undefined": the call was classified as object construction, so a
# throwaway self object was allocated and the base initialiser wrote to it
# instead of the real instance. It is now treated as a direct invocation of the
# base constructor with self passed explicitly.

class Base:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return self.name


class Child(Base):
    def __init__(self, name, age):
        Base.__init__(self, name)
        self.age = age


c = Child("bot", 3)
assert c.name == "bot"
assert c.age == 3
assert c.greet() == "bot"
