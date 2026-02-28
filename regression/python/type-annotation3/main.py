class Person:
    species = "Homo sapiens"

    def __init__(self, weight: int, age: int):
        self.weight = weight  # Instance variable
        self.age = age  # Instance variable

    def introduce(self) -> str:
        return "I am {self.age} years old and weigh {self.weight} kg"


person1 = Person(90, 25)
print(person1.introduce())
print(f"Species: {Person.species}")
