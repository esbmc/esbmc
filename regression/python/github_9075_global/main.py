count: int = 0


class Schema:

    def describe(self) -> str:
        return self.__class__.__name__


def bump() -> None:
    global count
    count += 1


bump()
assert Schema().describe() == "Schema"
assert count == 1
