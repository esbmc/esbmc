# esbmc/esbmc#7876 with a module-qualified container: `typing.List[int]` is an
# Attribute node, so its name is what marks the union as mixed.
import typing


def count(y: int | typing.List[int]) -> int:
    return len(y)


assert count([1, 2, 3]) == 3
