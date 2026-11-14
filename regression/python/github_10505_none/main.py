# #10505: len() of an Optional[C] call that returns a C calls C.__len__.
from typing import Optional


class C:

    def __len__(self) -> int:
        return 3


def make(b: bool) -> Optional[C]:
    if b:
        return None
    return C()


assert len(make(False)) == 3
