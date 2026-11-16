# GitHub #9839: the attribute is set only by an __init__ a later one replaces.
class Loc:
    def __getitem__(self, key: int) -> int:
        return 0


class Frame:
    def __init__(self) -> None:
        self.loc: Loc = Loc()

    def __init__(self) -> None:
        pass


fr = Frame()
assert fr.loc[0] == 0
