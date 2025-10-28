# Test case for GitHub issue #2966 (failure case)
# isinstance should correctly fail when condition is false
from typing import Any

i = 5  # i <= 10, so x will be int, not str
x: Any = "foo" if i > 10 else i + 10
assert isinstance(x, str)

