# len() of a computed instance whose class has a subclass (#10505)
from typing import Generic, TypeVar

T = TypeVar("T")


class C(Generic[T]):

    def __len__(self) -> int:
        return 1


class D(C[int]):

    def __len__(self) -> int:
        return 5


def make() -> C:
    return D()


assert len(make()) == 5
