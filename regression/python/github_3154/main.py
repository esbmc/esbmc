import re
from datetime import datetime


def foo(s: str | datetime) -> bool:
    if isinstance(s, datetime):
        return True

    assert isinstance(s, str)
    match = re.match(r"foo", s)
    return match is not None


s: str = "foo"
foo(s)
