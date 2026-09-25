def helper() -> int:
    import totally_missing_module
    return 1


def main() -> None:
    assert helper() == 1


main()
