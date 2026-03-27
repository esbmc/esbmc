from datetime import datetime


def foo(datetime_str: str | datetime) -> tuple[int, int, int]:
    if isinstance(datetime_str, datetime):
        return (
            datetime_str.year,
            datetime_str.month,
            datetime_str.day,
        )
    else:
        return (0, 0, 0)


dt: str | datetime
dt = datetime(2, 3, 4)
rv = (2, 4, 4)
assert foo(dt) == rv
