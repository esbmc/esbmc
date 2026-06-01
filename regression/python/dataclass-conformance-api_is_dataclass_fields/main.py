from dataclasses import dataclass, fields, is_dataclass


@dataclass
class User:
    name: str
    age: int


u = User("ana", 29)
assert is_dataclass(User)
assert is_dataclass(u)
assert len(fields(User)) == 2
