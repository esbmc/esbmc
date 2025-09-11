class Person:

    def __init__(self, name: int, age: int):
        self.name = name
        self.age = age

    def is_adult(self) -> bool:
        return self.age >= 18


person1 = Person(101, 25)
assert not person1.is_adult()
