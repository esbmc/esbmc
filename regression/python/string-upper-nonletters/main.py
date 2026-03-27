def main() -> None:
    assert "1234-_*?!".upper() == "1234-_*?!"
    assert "a1-b2_c3".upper() == "A1-B2_C3"

main()
