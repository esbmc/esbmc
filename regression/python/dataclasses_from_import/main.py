from dataclasses import dataclass


@dataclass
class Task:

    def __init__(self, value: int):
        self.value = value


task = Task(7)

assert task.value == 7
