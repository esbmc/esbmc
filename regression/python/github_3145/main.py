from datetime import datetime


def foo(d: datetime | str) -> None:
    pass


t = datetime(1, 1, 1, 0, 0, 0)
foo(t)
assert t.year == 1
assert t.month == 1
assert t.day == 1
