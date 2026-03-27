from typing import Any

class Foo:
    def __init__(self) -> None:
        pass

    def foo(self) -> int:
        return 42

class Bar:
    def __init__(self) -> None:
        pass

    def bar(self) -> int:
        return 17

def create(i: int) -> Any:
    if i == 0:
        return Foo()
    elif i == 1:
        return Bar()
    else:
        raise ValueError("Invalid value")

f = create(0)
f.bar()
