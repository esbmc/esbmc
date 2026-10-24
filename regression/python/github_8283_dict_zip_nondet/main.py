from esbmc import nondet_list


def main():
    a = nondet_list(3)
    b = nondet_list(3)
    d = dict(zip(a, b))
    assert len(d) <= len(a)


main()
