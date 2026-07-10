def main() -> None:
    # str(0.1234567) needs more than 6 fractional digits; CPython renders the
    # shortest round-trip repr "0.1234567". The frontend's fixed 6-decimal fold
    # rounds it to "0.123457", so pre-fix this false assertion verified
    # SUCCESSFUL (a soundness hole). The fold is now refused when it would lose
    # precision and a nondet string is emitted instead, so this must be a real
    # FAILED.
    assert str(0.1234567) == "0.123457"


main()
