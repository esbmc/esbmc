class Counter:
    def __init__(self):
        self.n = 0

    def tick(self) -> str:
        self.n += 1
        return "ok"


def main() -> None:
    c = Counter()
    label: int = c.tick()
    # Wrong: the call ran, so n is 1, not 0.
    assert c.n == 0


main()
