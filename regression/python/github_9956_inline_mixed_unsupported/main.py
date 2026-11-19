def mx():
    r = []
    r.append(2)
    r.append(1.5)
    return r


def main() -> None:
    assert sorted(mx())[0] == 1.5


main()
