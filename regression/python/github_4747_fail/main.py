NO_OVERRIDE = -1


class Cfg:

    def __init__(self, override: int = NO_OVERRIDE) -> None:
        self.override = override


def main() -> None:
    c = Cfg()
    assert c.override != NO_OVERRIDE  # must fail: default propagates correctly


main()
