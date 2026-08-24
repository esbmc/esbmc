def find_max(a, b):
    if a > b:
        return a
    return b


def main():
    result = find_max(10, 20)
    assert result >= 20


main()
