class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"

    # Constructor method
    def __init__(self, name:str, age:int):
        self.name = name    # Instance variable
        self.age = age      # Instance variable

    # Instance method
    def introduce(self) -> str:
        return "Hi, I'm {self.name} and I'm {self.age} years old"

    # Another instance method
    def have_birthday(self) -> int:
        self.age += 1
        return self.age

# Creating instances
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)
person1.have_birthday()
