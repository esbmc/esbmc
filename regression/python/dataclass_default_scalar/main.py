from dataclasses import dataclass


@dataclass
class Task:
    name: str
    priority: int = 5


t1 = Task("Morning Meeting")
t2 = Task("Standup", 1)

assert t1.name == "Morning Meeting"
assert t1.priority == 5
assert t2.name == "Standup"
assert t2.priority == 1
