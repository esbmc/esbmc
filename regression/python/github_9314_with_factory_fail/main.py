class C:

    def __enter__(self) -> "C":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False

    def get(self) -> int:
        return 1


def make() -> C:
    return C()


def make_ref() -> "C":
    return C()


with make() as f:
    x = f.get()
assert x == 1


def run() -> int:
    with make_ref() as g:
        return g.get() + 1


assert run() == 3
