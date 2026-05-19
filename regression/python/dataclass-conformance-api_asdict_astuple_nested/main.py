from dataclasses import asdict, astuple, dataclass


@dataclass
class Item:
    value: int


@dataclass
class Bag:
    tag: str
    count: int


b = Bag("test", 3)
i = Item(42)
assert b.tag == "test"
assert b.count == 3
assert i.value == 42

