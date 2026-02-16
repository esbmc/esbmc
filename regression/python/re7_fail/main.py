import re
from typing import Any

invalid_pattern: Any = 123
invalid_string: Any = 123

try:
    re.search(invalid_pattern, "abc")  # invalid at runtime, not static
except AssertionError:
    assert False

try:
    re.fullmatch("abc", invalid_string)  # invalid at runtime, not static
except AssertionError:
    assert False

