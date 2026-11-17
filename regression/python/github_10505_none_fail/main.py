# #10505: len() of a call that may return None raises TypeError.
from typing import Optional


class C:

    def __len__(self) -> int:
        return 3


def make(b: bool) -> Optional[C]:
    if b:
        return None
    return C()


x = len(make(True))
