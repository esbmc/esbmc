def main() -> None:
    # Pre-fix, an empty "{}" field formatted 1.0 as "1" (dropping the ".0"),
    # so this false assertion verified SUCCESSFUL. CPython gives "1.0", so it
    # must be a real FAILED.
    assert "{}".format(1.0) == "1"


main()
