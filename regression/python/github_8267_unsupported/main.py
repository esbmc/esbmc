class C:
    v = 0


def main():
    c = C()
    if nondet_bool():
        C.v = 2
    else:
        c.v = 5
    assert c.v != 0


main()
