def f():
    return 1


def main():
    ls = [f]
    ls2 = ls[0:1]
    c = ls2[0]()
    assert c == 3


main()
