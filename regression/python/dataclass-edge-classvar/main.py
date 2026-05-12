from dataclasses import dataclass, fields
from typing import ClassVar


@dataclass
class Config:
    VERSION: ClassVar[int] = 1
    name: str


c = Config("alpha")
assert c.name == "alpha"
assert Config.VERSION == 1
assert len(fields(Config)) == 1
