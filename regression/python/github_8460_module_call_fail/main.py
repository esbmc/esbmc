import math
import os
import string


def letters(text: str) -> bool:
    return string.ascii_letters.startswith(text)


def root(x: float) -> float:
    return math.sqrt(x)


def local_import(x: float) -> float:
    import math
    return math.sqrt(x)


def f(string: str) -> int:
    return len(string.split())


assert letters("abc")
assert root(4.0) == 2.0
assert local_import(9.0) == 3.0
assert not os.path.exists("no_such_dir/x")
assert f("a b") == 3
