def main() -> None:
    assert "a b\tc".rfind("b\t") == 2
    assert "a b\tc".rfind(" ") == 1

main()
