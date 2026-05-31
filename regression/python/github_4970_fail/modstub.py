class C:
    def __init__(self, v: int):
        self.v = v

    def check(self) -> None:
        assert self.v == TAG      # module global, referenced inside a method


TAG: int = 2                      # defined AFTER the class; value != the v below
