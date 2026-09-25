import threading

# Two function-scope subclass instances write the same module-level
# global without synchronisation. The hoist that makes each local
# binding visible to its trampoline must not lose the race.
shared: int = 0


class Bumper(threading.Thread):
    def __init__(self, step: int) -> None:
        super().__init__()
        self.step: int = step

    def run(self) -> None:
        global shared
        shared = shared + self.step


def race() -> None:
    b1: Bumper = Bumper(1)
    b2: Bumper = Bumper(2)
    b1.start()
    b2.start()
    b1.join()
    b2.join()


race()
