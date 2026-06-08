class Worker:
    def run(self) -> int:
        return 9


class Holder:
    def __init__(self):
        self.worker = Worker()

    def execute(self) -> int:
        return self.worker.run()


h = Holder()

assert h.execute() == 9
