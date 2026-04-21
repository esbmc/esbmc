# example_13_sleep.py
import time

class Task:
    def __init__(self, name: str, duration: int):
        self.name = name
        self.duration = duration
        self.completed = False
        self.start_time = 0.0
        self.end_time = 0.0

    def execute(self):
        print("Starting task:", self.name)
        self.start_time = time.time()

        # Simulate work with sleep
        time.sleep(self.duration)

        self.end_time = time.time()
        self.completed = True
        print("Completed task:", self.name)

        actual_duration = self.end_time - self.start_time
        print("Actual duration:", actual_duration, "seconds")

def demonstrate_sleep():
    tasks = [
        Task("Quick Task", 1),
        Task("Medium Task", 2),
        Task("Long Task", 3)
    ]

    print("Starting task execution")

    for task in tasks:
        task.execute()
        status = "Completed" if task.completed else "Pending"
        print("Task status:", status, "\n")

    print("All tasks completed")

if __name__ == "__main__":
    demonstrate_sleep()
