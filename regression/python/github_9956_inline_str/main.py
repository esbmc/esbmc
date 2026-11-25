def ms():
    r = []
    r.append("b")
    r.append("a")
    return r


def main() -> None:
    assert sorted(ms())[0] == "a"
    assert max(ms()) == "b"


main()
