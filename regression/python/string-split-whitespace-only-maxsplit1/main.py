def main() -> None:
    s = " \t  \n"
    parts = s.split(None, 1)
    assert len(parts) == 0

main()
