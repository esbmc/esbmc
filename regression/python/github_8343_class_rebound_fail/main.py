# As github_8343_class_rebound, with a wrong total.
class Count:

    def __init__(self, n: int):
        self.n = n


def pair(n: int) -> list[int]:
    return [7, 8]


Count = pair


def main():
    total = 0
    for x in Count(3):
        total += x
    assert total == 14


main()
