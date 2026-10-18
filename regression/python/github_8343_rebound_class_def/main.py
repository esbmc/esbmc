class Count:
    def __init__(self, n: int):
        self.n = n


def pair(n: int) -> list[int]:
    return [7, 8]


def Count(n: int) -> list[int]:
    return [7, 8]


def main():
    total = 0
    for x in Count(3):
        total += x
    assert total == 15


main()
