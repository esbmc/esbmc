from dataclasses import dataclass, field


def make_counter() -> int:
    return 100


@dataclass
class Counter:
    name: str
    value: int = field(default_factory=make_counter)


# Each instance must get its own fresh value from the factory call.
c1 = Counter("a")
c2 = Counter("b")

assert c1.value == 100
assert c2.value == 100

c1.value += 1
# Mutating one instance must not affect the other (per-instance evaluation).
assert c1.value == 101
assert c2.value == 100
