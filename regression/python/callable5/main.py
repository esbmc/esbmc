from typing import Callable

g: Callable[[int], int] = lambda x: x + 3

assert g(2) == 5
