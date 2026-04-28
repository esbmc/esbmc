from dataclasses import InitVar, dataclass


@dataclass
class Task:
    name: str
    seed: InitVar[int]
    priority: int = 0

    def __post_init__(self, seed) -> None:
        self.priority = seed + len(self.name)


task = Task("abc", 4)

assert task.priority == 7
assert not hasattr(task, "seed")