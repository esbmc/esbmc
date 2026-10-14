def main() -> None:
    # startswith/endswith accept a tuple of affixes: True if ANY matches.
    assert "abc".startswith(("x", "ab"))
    assert "hello".startswith(("he", "z"))
    assert not "abc".startswith(("x", "y"))

    assert "main.py".endswith((".txt", ".py"))
    assert not "readme.md".endswith((".txt", ".py"))

    # Single-string forms are unaffected.
    assert "abc".startswith("ab")
    assert "abc".endswith("bc")


main()
