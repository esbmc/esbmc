from dataclasses import dataclass, fields, is_dataclass, InitVar
from typing import ClassVar


@dataclass
class C:
    x: int
    y: ClassVar[int] = 10
    z: InitVar[int] = 0


c = C(1, 2)

assert is_dataclass(C) == True
assert is_dataclass(c) == True
assert is_dataclass(123) == False

fs = fields(C)
assert len(fs) == 1
assert fs[0].name == "x"
