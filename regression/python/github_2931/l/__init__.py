from typing import Any
import l.md as md

class Foo:
    def __init__(self) -> None:
        self._md: md.Bar = md.Bar('bar')

def create(s: str) -> Any:
    if s == "foo":
        return Foo()
    else:
        assert False, "Invalid string"
