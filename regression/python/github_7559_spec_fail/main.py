# A format spec pads the repr text, which is not modelled; the part must stay
# unconstrained rather than fold to the unpadded repr (#7559).
def main() -> None:
    assert f"{42!r:>6}" == "42"


main()
