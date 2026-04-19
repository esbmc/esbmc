from typing import Union
def returns_union(b: bool) -> Union[int, bool]:
    if b:
        return 1
    else:
        return False

assert returns_union(True) != 1
assert returns_union(False) != False
assert returns_union(5) != 1
assert returns_union([1]) != 1
