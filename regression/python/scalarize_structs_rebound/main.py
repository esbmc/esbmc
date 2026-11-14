# A tuple variable rebound to an int is assigned a value that is not a struct;
# --scalarize-structs (#7521) leaves it whole instead of splitting the int.
from typing import Any

x: Any = (1, 2)
y = x[0]
i = 0
while i < 3:
    x = 5
    i = i + 1
assert y == 1
