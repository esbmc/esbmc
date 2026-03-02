class Person:

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        return "Hello, I'm {self.name} and I'm {self.age} years old."

    def have_birthday(self):
        self.age += 1
        return "Happy birthday! I'm now {self.age} years old."

    def is_adult(self) -> bool:
        return self.age >= 18
