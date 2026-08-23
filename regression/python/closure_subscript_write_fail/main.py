def counter():
    c = [0]

    def bump():
        c[0] += 1

    bump()
    bump()
    return c[0]


def main():
    assert counter() == 3


main()
