# Each class name is rebound before any read, by an assignment and by a
# def, so a call of it calls the function.
class Count:

    def __init__(self, n: int):
        self.n = n


class Pair:
    pass


def pair(n: int) -> list[int]:
    return [7, 8]


Count = pair


def Pair(n: int) -> list[int]:
    return [n, n]


def main():
    total = 0
    for x in Count(3):
        total += x
    assert total == 15
    for y in Pair(2):
        total += y
    assert total == 19


main()
