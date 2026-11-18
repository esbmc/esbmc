def mf():
    r = []
    r.append(2.5)
    r.append(1.5)
    return r


def main() -> None:
    assert sorted(mf())[0] < sorted(mf())[1]
    assert sorted(mf())[0] == 1.5
    assert max(mf()) == 2.5
    assert min(mf()) == 1.5
    assert sum(mf()) == 4.0


main()
