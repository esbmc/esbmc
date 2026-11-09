"""A function value called at two argument types (#9797)."""
import math

LIMIT = 10


class Box:
    size = 1

    def get(self) -> int:
        return self.size


def add(a, b):
    return a + b


def call(f, a, b):
    return f(a, b)


assert call(add, 1, 2) == 3
assert call(add, "a", "b") == "ab"
assert Box().get() + LIMIT == 11
assert math.floor(1.5) == 1
