import math

x: int = -1

try:
    result: float = math.sqrt(x)  # Domain error!
except ValueError:
    assert False, "negative float input"
