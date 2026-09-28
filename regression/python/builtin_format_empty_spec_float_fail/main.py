def main() -> None:
    # The empty-spec float fold is now active, so this false assertion is a real
    # FAILED: format(-1.0) folds to str(-1.0) == "-1.0", not "-1". Pre-fix this
    # call errored ("format() spec '' is not supported for this value") rather
    # than folding at all.
    assert format(-1.0) == "-1"


main()
