from dataclasses import dataclass, field


@dataclass
class Counter:
    name: str
    count: int = field(init=False, default=0)


c = Counter("hits")
assert c.name == "hits"
assert c.count == 0
