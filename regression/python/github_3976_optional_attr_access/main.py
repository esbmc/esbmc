from typing import Optional


class Box:
    def __init__(self, value: int):
        self.value = value


def get_value(flag: bool) -> int:
    box: Optional[Box] = None
    if flag:
        box = Box(7)
    if box:
        return box.value
    return 0


get_value(True)
get_value(False)
assert True
