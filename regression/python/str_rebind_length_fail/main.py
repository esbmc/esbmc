def main() -> None:
    k = "abc"
    assert len(k) == 3
    assert k[2] == "b"
    k = "z"
    assert len(k) == 1
    assert k == "z"
    k = "hello"
    assert k[1:3] == "el"
    assert len(k) == 5


main()
