# Test TypedDict with nested types - fail case
from typing import TypedDict, List, Optional


class Task(TypedDict):
    title: str
    tags: List[str]
    parent: Optional[int]


def process_task(task: Task) -> None:
    pass


t: dict = {"title": "test", "tags": ["a", "b"], "parent": None}
process_task(t)
x: int = 1
assert x == 0  # Should fail
