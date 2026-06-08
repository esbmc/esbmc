class Worker:
    def run(self) -> int:
        return 9

class Host:
    def __init__(self) -> None:
        self.worker = Worker()

    def run_worker(self) -> int:
        return self.worker.run()

h = Host()
assert h.run_worker() == 99  # wrong value, should be 9
