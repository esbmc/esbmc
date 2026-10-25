class Job:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def stop(self) -> int:
        return self.n


def main() -> None:
    jobs: dict[str, Job] = {"a": Job(3)}
    assert jobs.pop("a").stop() == 3
    assert len(jobs) == 0
    j = Job(4)
    jobs["b"] = j
    j.n = 5
    assert jobs.pop("b").stop() == 5


main()
