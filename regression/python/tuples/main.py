# example_11_tuples.py
from typing import Tuple, List
from dataclasses import dataclass

# Type aliases using tuples
Coordinate = Tuple[int, int]
TimeRange = Tuple[int, int]

@dataclass
class Task:
    name: str
    id: int

# Complex type combining tuple with custom class
TaskInstance = Tuple[int, Task]
TaskSchedule = List[TaskInstance]

def create_coordinate(x: int, y: int) -> Coordinate:
    return (x, y)

def create_time_range(start: int, end: int) -> TimeRange:
    if end <= start:
        raise ValueError("End time must be after start time")
    return (start, end)

def schedule_task(time: int, task: Task) -> TaskInstance:
    return (time, task)

def demonstrate_tuples():
    # Basic coordinate tuple
    coord: Coordinate = create_coordinate(10, 20)
    print(f"Coordinate: x={coord[0]}, y={coord[1]}")

    # Time range tuple
    try:
        work_hours: TimeRange = create_time_range(9, 17)
        print(f"Work hours: {work_hours[0]}:00 to {work_hours[1]}:00")
    except ValueError as e:
        print(f"Error: {e}")

    # Task scheduling with tuples
    task1 = Task("Morning Meeting", 1)
    task2 = Task("Lunch Break", 2)
    task3 = Task("Project Review", 3)

    schedule: TaskSchedule = [
        schedule_task(9, task1),
        schedule_task(12, task2),
        schedule_task(14, task3)
    ]

    print("\nDaily Schedule:")
    for time, task in schedule:
        print(f"{time}:00 - {task.name} (ID: {task.id})")

    # Tuple unpacking
    first_task_time, first_task = schedule[0]
    print(f"\nFirst task of the day: {first_task.name} at {first_task_time}:00")

if __name__ == "__main__":
    demonstrate_tuples()

