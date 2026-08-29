def tail(n):
    if n > 0:
        r = tail(n - 1)
        return r
    return 0


def main():
    assert tail(3) == 7


main()
