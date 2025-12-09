from typing import Union
def returns_union(b: bool) -> Union[int, bool]:
    if b:
        return 1
    else:
        return False

assert returns_union(0.0) == False
assert returns_union(0.1) == 1
assert returns_union(-0.5) == 1
assert returns_union(5.0) == 1
