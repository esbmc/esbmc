def classify(n: int) -> int:
    values = [1, 2, 3]
    if n > 0 and n < 0:
        return values[0]
    return values[1]


def main():
    assert classify(5) == 2


if __name__ == "__main__":
    main()
