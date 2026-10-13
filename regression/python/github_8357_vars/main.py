# Issue #8357: vars() reaches Outer by name, whose make then runs on an Other,
# so `self.Inner` is left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


assert vars()["Out" + "er"].make(Other(), 4) == 5
