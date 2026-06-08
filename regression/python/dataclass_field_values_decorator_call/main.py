from dataclasses import dataclass


@dataclass()
class Task:
    name: str
    id: int


task = Task("Morning Meeting", 1)

assert task.id == 1
assert task.name == "Morning Meeting"
