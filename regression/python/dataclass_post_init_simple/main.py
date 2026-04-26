from dataclasses import dataclass


@dataclass
class Counter:
    base: int
    value: int = 0

    def __post_init__(self) -> None:
        self.value = self.base + 1


c = Counter(10)

assert c.value == 11
