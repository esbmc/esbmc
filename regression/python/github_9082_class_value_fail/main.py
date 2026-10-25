class Job:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def stop(self) -> int:
        return self.n


def main() -> None:
    jobs: dict[str, Job] = {"a": Job(1)}
    j = jobs["a"]
    assert j.stop() == 2


main()
