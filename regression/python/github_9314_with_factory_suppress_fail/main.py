class C:

    def __init__(self) -> None:
        self.exited = False

    def __enter__(self) -> "C":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        self.exited = True
        return True


def make() -> C:
    return C()


with make() as f:
    raise ValueError("suppressed")
assert not f.exited
