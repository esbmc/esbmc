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

b = nondet_bool()

if b:
    dt1:datetime = datetime(2, 3, 4)
    rv = (2, 3, 4)
    assert foo(dt1) == rv
else:
    dt2:str = "foo"
    rv = (0, 0, 0)
    assert foo(dt2) == rv

assert b or isinstance(dt2, str)
assert not b or isinstance(dt1, datetime)
