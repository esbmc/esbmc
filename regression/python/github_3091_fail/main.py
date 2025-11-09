from typing import Any, Literal, overload

class Foo:
    def __init__(self) -> None:
        pass

    def foo(self) -> None:
        pass
    

class Bar:
    def __init__(self) -> None:
        pass

    def bar(self) -> None:
        pass

@overload
def create(s: Literal["foo"], t: str) -> Foo:
    ...

@overload
def create(s: Literal["bar"], t: str) -> Bar:
    ...
    
def create(s: str, t: str) -> Foo | Bar:
    if s == 'foo':
        return Foo()
    elif s == 'bar':
        return Bar()
    else:
        raise NotImplementedError("Unknown class")

b = create("bar", t='t')
b.bar()
assert not isinstance(b, Bar)
