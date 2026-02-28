from typing import Any

a: Any = None
a = []  # OK
a = 2  # OK
assert a == 2  # OK
s: str = ''
s = a  # OK
assert s == 2  # OK
