# GitHub #9839: an attribute set on some paths is not dispatched on.
class Loc:
    def __getitem__(self, key: int) -> int:
        return 0


class Frame:
    def __init__(self, c: bool) -> None:
        if c:
            self.loc: Loc = Loc()


fr = Frame(False)
assert fr.loc[0] == 0
