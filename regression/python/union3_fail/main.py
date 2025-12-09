from typing import Union
def returns_union(b: bool) -> Union[int, bool]:
    if b:
        return 1
    else:
        return False

# Wrong expectation for float 0.0
assert returns_union(0.0) == 1

