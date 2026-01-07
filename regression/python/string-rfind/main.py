def main() -> None:
    assert "banana".rfind("na") == 4
    assert "banana".rfind("ba") == 0
    assert "banana".rfind("x") == -1
    assert "aaaa".rfind("aa") == 2
    assert "edge".rfind("") == 4

main()
