import random


def is_str(x: str) -> bool:
    return isinstance(x, str)


s = "a" if random.random() < 0.5 else None
assert is_str(s) == (s is not None)
