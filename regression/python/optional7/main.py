from typing import Optional
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None

    def start(self, time: int):
        self.start_time = time

    def end(self, time: int):
        self.end_time = time

    def duration(self) -> Optional[int]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

def demonstrate_optional():
    # Create a task without times
    task1 = Task("First Task")
    print("New task:", task1)

    # Start the task
    task1.start(100)
    print("After starting:", task1)

    # End the task
    task1.end(150)
    print("After ending:", task1)
    print("Task duration:", task1.duration())

    # Demonstrate None handling
    task2 = Task("Second Task")
    print("\nTask 2 duration:", task2.duration())

    # Safe access pattern
    if task2.start_time is not None:
        print("Task 2 started at:", task2.start_time)
    else:
        print("Task 2 hasn't started yet")

if __name__ == "__main__":
    demonstrate_optional()
