# Test case for GitHub issue #2966
# isinstance should work correctly with ternary operators
from typing import Any

i = 42
x: Any = "foo" if i > 10 else i + 10
assert isinstance(x, str)

