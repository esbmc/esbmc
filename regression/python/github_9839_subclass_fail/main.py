# GitHub #9839: a subclass instance stored in the attribute keeps its own
# __getitem__, so the attribute's static class is not dispatched on.
class Loc:
    def __getitem__(self, key: int) -> int:
        return 0


class SubLoc(Loc):
    def __getitem__(self, key: int) -> int:
        return 1


class Frame:
    def __init__(self, l: Loc) -> None:
        self.loc: Loc = l


fr = Frame(SubLoc())
assert fr.loc[0] == 0
