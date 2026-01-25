# Test TypedDict with multiple fields - fail case
from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int


def greet(person: Person) -> None:
    pass


p: dict = {"name": "Alice", "age": 30}
greet(p)
assert False  # Should fail
