from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Config:
    version: ClassVar[int] = 3
    name: str


cfg = Config("prod")

assert Config.version == 3
assert cfg.name == "prod"