def main() -> None:
    taken: int = 0
    try:
        import json
        taken = 1
    except ImportError:
        taken = 2
    assert taken == 1


main()
