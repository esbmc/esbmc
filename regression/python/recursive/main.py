# example_4_recursive.py
from typing import List

class Task:
    def __init__(self, name: str):
        self.name = name
        self.subtasks: List[Task] = []

    def add_subtask(self, task: 'Task'):
        self.subtasks.append(task)

    def __str__(self):
        result = f"Task({self.name})"
        if self.subtasks:
            subtasks_str = ", ".join(str(task) for task in self.subtasks)
            result += f" with subtasks: [{subtasks_str}]"
        return result

def demonstrate_recursive_tasks():
    # Create main task
    main_task = Task("Main")

    # Add some subtasks
    sub1 = Task("Subtask 1")
    sub2 = Task("Subtask 2")
    main_task.add_subtask(sub1)
    main_task.add_subtask(sub2)

    # Add a sub-subtask
    sub_sub = Task("Sub-subtask")
    sub1.add_subtask(sub_sub)


if __name__ == "__main__":
    demonstrate_recursive_tasks()
