# Operational model for collections module

from typing import Any, Optional


def defaultdict(default_factory: Optional[Any] = None, *args, **kwargs) -> dict:
    """Create a defaultdict - modeled as a plain dict for verification purposes.

    Approximations:
    - The default_factory is tracked by the preprocessor and used to insert
      missing-key defaults; it is not stored on the dict object itself.
    - Initial data passed as a positional mapping or iterable (*args) and any
      keyword arguments (**kwargs) are accepted but silently ignored. Pre-populated
      defaultdicts are not modeled; only keys written explicitly in the program
      will be present in the verification model.
    """
    return {}


class Counter:

    def __init__(self) -> None:
        self.data: dict[tuple[int, int], int]
        self.data = {}
        self.count: int = 0

    def __getitem__(self, key: tuple[int, int]) -> int:
        try:
            return self.data[key]
        except KeyError:
            return 0

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        try:
            self.data[key]
        except KeyError:
            self.count = self.count + 1
        self.data[key] = value

    def values(self) -> list[int]:
        return self.data.values()

    def __bool__(self) -> bool:
        return self.count != 0
