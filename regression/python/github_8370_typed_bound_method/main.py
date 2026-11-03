from typing import Callable

class B:
    def __init__(self) -> None:
        self.child: Callable[[], int] = self.func2

    def func2(self) -> int:
        return 42

b = B()
assert b.child() == 42
