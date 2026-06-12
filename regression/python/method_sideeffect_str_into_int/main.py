class Counter:
    def __init__(self):
        self.n = 0

    def tick(self) -> str:
        # Side effect: mutate self while returning a str.
        self.n += 1
        return "ok"


def main() -> None:
    c = Counter()
    # The str return does not fit the int annotation, so the result is dropped,
    # but the call (and its side effect on self.n) must still execute.
    label: int = c.tick()
    assert c.n == 1


main()
