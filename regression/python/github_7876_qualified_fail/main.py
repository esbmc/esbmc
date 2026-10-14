# Counterpart to github_7876_qualified.
import typing


def count(y: int | typing.List[int]) -> int:
    return len(y)


assert count([1, 2, 3]) != 3
