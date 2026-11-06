# GitHub #9839: obj.attr[key] calls the __getitem__ of attr's class.
class Loc:
    def __getitem__(self, key) -> int:
        return 0


class Frame:
    def __init__(self) -> None:
        self.loc: Loc = Loc()


fr = Frame()
assert fr.loc[0, 1] == 1
