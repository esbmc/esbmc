class C:
    def me(self) -> "C":
        return self

    def get(self) -> int:
        return 1


def make() -> C:
    return C()
