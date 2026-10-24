class Schema:
    def __init__(self, n: int):
        self.n = n

    def describe(self) -> str:
        return self.__class__.__name__ + str(10 // self.n)


def main() -> None:
    Schema(0).describe()


main()
