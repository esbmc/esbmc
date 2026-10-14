def counter():
    c = [0]

    def bump():
        c[0] += 1

    bump()
    bump()
    return c[0]


def nested_dict():
    d = {"n": 1}

    def set_it():
        d["n"] = 42

    set_it()
    return d["n"]


def deep():
    box = [1]

    def mid():
        def inner():
            box[0] = 7

        inner()

    mid()
    return box[0]


def main():
    assert counter() == 2
    assert nested_dict() == 42
    assert deep() == 7


main()
