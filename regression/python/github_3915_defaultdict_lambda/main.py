from collections import defaultdict


def main():
    d = defaultdict(lambda: 0)
    d[5] = 7
    assert d[5] == 7


main()
