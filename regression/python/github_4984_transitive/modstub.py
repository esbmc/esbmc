def base() -> int:
    return TAG                    # helper reads the module global


class C:
    def __init__(self, v: int):
        self.v = v

    def check(self) -> None:
        assert self.v == base()   # method calls a sibling helper, not TAG directly


TAG: int = 7                      # reachable from C only transitively (C -> base -> TAG)
