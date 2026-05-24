class CMCond:
    def __init__(self, swallow: bool):
        self.swallow = swallow
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb) -> bool:
        return self.swallow


def main() -> None:
    suppressed = False
    try:
        with CMCond(False):
            raise ValueError("x")
        suppressed = True
    except ValueError:
        pass
    assert suppressed
main()
