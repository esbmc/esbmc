def main():
    a = [1, 2]
    b = [3, 4]
    d = dict(zip(a, b))
    assert d[1] == 4


main()
