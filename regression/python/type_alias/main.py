# example_10_type_aliases.py
from typing import NewType

# Simple type alias
Time = int

# More specific type using NewType
TaskId = NewType('TaskId', int)
Priority = NewType('Priority', int)

class Task:
    def __init__(self, id: TaskId, priority: Priority):
        self.id = id
        self.priority = priority

def send_task(time: Time, task: Task) -> None:
    print(f"Sending task {task.id} at time {time} with priority {task.priority}")

def process_at_time(time: Time) -> None:
    print(f"Processing at timestamp: {time}")

def demonstrate_type_aliases():
    # Using Time alias
    current_time: Time = 1000
    process_at_time(current_time)

    # Using TaskId and Priority
    task_id = TaskId(1)
    priority = Priority(2)
    task = Task(task_id, priority)

    # Send task
    send_task(current_time, task)

    # Demonstrate type safety
    # This would raise a type error in a type checker:
    # wrong_id = TaskId("1")  # Error: Expected int, got str

    # But this works:
    next_time: Time = current_time + 100
    print(f"Next processing time: {next_time}")

if __name__ == "__main__":
    demonstrate_type_aliases()
