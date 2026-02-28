class Person:

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


people = [Person("Alice", 25), Person("Bob", 30), Person("Charlie", 22)]

# Check if a person named 'Alice' exists
assert any(p.name == "Alice" for p in people), "Alice is missing!"

# Verify the maximum age
assert max(p.age for p in people) == 30, "Max age is incorrect!"
