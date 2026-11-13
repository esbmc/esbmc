def dec(f):
    return 42


@dec
def g():
    return "g"


def main():
    try:
        g()
        assert False
    except TypeError:
        pass


main()
