def apply(g, v: int) -> int:
    return g(v)


def main():
    n: int = 10
    inc = lambda x: x + n
    assert apply(inc, 5) == 15


main()
