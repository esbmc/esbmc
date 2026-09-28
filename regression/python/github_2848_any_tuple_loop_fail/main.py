from typing import Any

x: Any = (1, 2)
y = x[0]
i = 0
while i < 3:
    x = 5
    i = i + 1
assert y == 2
