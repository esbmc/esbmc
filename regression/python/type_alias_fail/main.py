from typing import NewType

TaskId = NewType('TaskId', int)
Priority = NewType('Priority', int)
Time = int

def make_id(value: int) -> TaskId:
    return TaskId(value)

def main() -> None:
    task_id: TaskId = make_id(7)
    priority: Priority = Priority(3)
    current_time: Time = 100

    total = task_id + priority + current_time
    assert total == 110

    assert task_id == 8

main()
