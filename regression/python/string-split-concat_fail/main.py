def main() -> None:
    # Unsoundness variant from issue #5085: the last part is "com", so the
    # assertion is FALSE and ESBMC must report a counterexample.
    s = "x." + "com"
    assert s.split(".")[-1] != "com"


main()
