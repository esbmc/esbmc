def main() -> None:
    # "%.2f" % 3.14159 is "3.14", not "3.142".
    assert "%.2f" % 3.14159 == "3.142"


main()
