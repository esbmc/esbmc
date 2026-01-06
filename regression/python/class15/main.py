from datetime import datetime

def foo(dt: datetime) -> int:
    return dt.year

assert foo(datetime(2, 3, 4)) == 2

