from dataclasses import dataclass, is_dataclass


@dataclass
class Animal:
    name: str


@dataclass
class Dog(Animal):
    breed: str


a = Animal("cat")
d = Dog("Rex", "Labrador")
assert is_dataclass(Animal)
assert is_dataclass(Dog)
assert is_dataclass(a)
assert is_dataclass(d)
