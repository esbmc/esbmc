from dataclasses import dataclass


@dataclass
class Animal:
    name: str
    legs: int


@dataclass
class Dog(Animal):
    breed: str


d = Dog("Rex", 4, "Labrador")
assert d.name == "Rex"
assert d.legs == 4
assert d.breed == "Labrador"
