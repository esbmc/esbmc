from typing import Any

class Foo:
    def __init__(self) -> None:
        pass

    def foo(self, x: int) -> int:
        return x + 1

class Bar:
    def __init__(self) -> None:
        pass
        
    def bar(self, x: str) -> bool:
       return x == "bar"

def create(t: str) -> Any:
    if t == "foo":
        return Foo()
    elif t == "bar":
        return Bar()
    else:
        raise Exception("Unknown type")
