# GitHub #9839: an attribute __init__ may return before setting is not
# dispatched on.
class Loc:
    def __getitem__(self, key: int) -> int:
        return 0


class Frame:
    def __init__(self, c: bool) -> None:
        if c:
            return
        self.loc: Loc = Loc()


fr = Frame(True)
assert fr.loc[0] == 0
