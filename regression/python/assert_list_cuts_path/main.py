# assert on an empty list raises AssertionError, so l[0] never runs on it.
def main():
    l: list[int] = []
    if nondet_bool():
        l.append(1)
    assert l
    x = l[0]


main()
