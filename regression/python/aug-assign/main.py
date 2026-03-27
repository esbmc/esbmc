class Counter:

    def __init__(self):
        self.value = 1


def main() -> None:
    counts = [0, 3, 5]
    counts[0] += 1
    counts[1] -= 1
    counts[2] *= 2
    counts[0] += 2
    counts[0] //= 3
    counts[1] %= 2
    assert counts[0] == 1
    assert counts[1] == 0
    assert counts[2] == 10

    bucket = {0: 1}
    bucket[0] |= 2
    bucket[0] ^= 1
    bucket[0] <<= 1
    bucket[0] >>= 1
    assert bucket[0] == 2

    counter = Counter()
    counter.value += 4
    assert counter.value == 5


main()
