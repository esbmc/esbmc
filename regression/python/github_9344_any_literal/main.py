from typing import Any
flag = nondet_bool()
x: Any = "foo" if flag else 5
assert isinstance(x, str) == flag
assert isinstance(x, int) == (not flag)
