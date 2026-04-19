class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    # Constructor method
    def __init__(self, weight: int, age: int):
        self.weight = weight    # Instance variable
        self.age = age          # Instance variable
    
    # Instance method
    def gain_weight(self) -> int:
        self.weight += 1
        return self.weight
    
    # Another instance method
    def have_birthday(self) -> int:
        self.age += 1
        return self.age
    
    # Additional method for better functionality
    def introduce(self) -> str:
        return "I am {self.age} years old and weigh {self.weight} kg"

# Creating instances
person1 = Person(90, 25)
person1.gain_weight()
person1.have_birthday()

# Test the functionality
assert person1.weight == 91
assert person1.age == 25

# Additional validation
print(person1.introduce())
print(f"Species: {Person.species}")

# Create another person for comparison
person2 = Person(75, 30)
print(f"Person 2: {person2.introduce()}")
