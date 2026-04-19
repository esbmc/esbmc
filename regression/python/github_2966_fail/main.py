from typing import Any
i = 5
x: Any = "foo" if i > 10 else i + 10
assert isinstance(x, str)
