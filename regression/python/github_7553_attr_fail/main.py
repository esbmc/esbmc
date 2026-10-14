# A user attribute called `keys` is not a dict view. It reaches the list
# comparison as a `keys` member too, so matching on the member name alone
# would wrongly prove this one (#7553).
class Node:
    def __init__(self) -> None:
        self.keys = [1]


def main() -> None:
    n = Node()
    assert n.keys != [1]


main()
