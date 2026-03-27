from datetime import datetime


def foo(datetime_str: str | datetime) -> tuple[int, int, int]:
    if isinstance(datetime_str, int):
        return (
            datetime_str.year,
            datetime_str.month,
            datetime_str.day,
        )
    else:
        return (0, 0, 0)


assert foo("foo") == (0, 0, 0)
