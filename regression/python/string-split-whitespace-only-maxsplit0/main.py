def main() -> None:
    s = " \t  \n"
    parts = s.split(None, 0)
    assert len(parts) == 0

main()
