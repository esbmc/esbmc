from typing import List


# Test typed list as class attribute with external method calls
class TaskQueue:

    def __init__(self):
        self.tasks: List[str] = []

    def size(self) -> int:
        return len(self.tasks)


queue = TaskQueue()
queue.tasks.append("task1")
queue.tasks.append("task2")

assert queue.size() == 2
assert "task1" in queue.tasks
assert "task2" in queue.tasks
