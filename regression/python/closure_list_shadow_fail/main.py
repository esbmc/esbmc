def shadowed_list():
    c = [0]

    def inner():
        c = [9]
        return c[0]

    inner()
    return c[0]


def main():
    assert shadowed_list() == 9


main()
