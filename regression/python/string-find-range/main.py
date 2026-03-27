def main() -> None:
    assert "banana".find("na", 3) == 4
    assert "banana".find("na", 0, 4) == 2
    assert "banana".rfind("na", 0, 4) == 2


main()
