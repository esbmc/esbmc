from dataclasses import dataclass
from typing import Optional


@dataclass(repr=False)
class Config:
    name: str
    retries: int = 0
    tag: Optional[str] = None


c1 = Config("fast")
c2 = Config("slow", 5, "v2")
assert c1.name == "fast"
assert c1.retries == 0
assert c2.retries == 5
assert c2.tag == "v2"
