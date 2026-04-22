from typing import List

class Task:
    def __init__(self, name: str):
        self.name = name
        self.subtasks: List[Task] = []

    def add_subtask(self, task: "Task"):
        self.subtasks.append(task)

    def __str__(self):
        return f"Task({self.name})"

def build_joined() -> str:
    root = Task("Root")
    root.add_subtask(Task("A"))
    root.add_subtask(Task("B"))
    return " | ".join(str(task) for task in root.subtasks)

result = build_joined()
