from typing import Optional


class Box:
    def __init__(self, value: int):
        self.value = value


box: Optional[Box] = None
x = nondet_bool()
if x:
    box = Box(7)
if box is not None:
    assert box.value == 7
else:
    assert not x
