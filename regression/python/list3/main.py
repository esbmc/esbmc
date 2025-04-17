# Membership Operators
lst = [1, 2, 3, 4, 5]
assert a in lst
assert c not in lst


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [
    Person("Alice", 25),
    Person("Bob", 30),
    Person("Charlie", 22)
]

# Check if a person named 'Alice' exists
assert any(p.name == "Alice" for p in people), "Alice is missing!"

# Verify the maximum age
assert max(p.age for p in people) == 30, "Max age is incorrect!"
