class Job:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def stop(self) -> int:
        return self.n


def main() -> None:
    a = Job(1)
    jobs: dict[str, Job] = {"a": a}
    a.n = 2
    assert jobs["a"].n == 2
    jobs["c"] = Job(3)
    c = jobs["c"]
    assert c.stop() == 3
    p = jobs.pop("c")
    assert p.stop() == 3
    k, v = jobs.popitem()
    assert k == "a" and v.stop() == 2


main()
