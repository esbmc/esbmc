# A nested def that shadows the alias keeps its own call (#8352).
class A:
    def m(self) -> int:
        return 1


cls = A


def run() -> int:
    def cls() -> int:
        return 2

    return cls()


assert run() == 2
