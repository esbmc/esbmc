def main() -> None:
    # Pre-fix, the consteval fold treated digits as word-internal (alnum
    # boundaries), folding "3d movie".title() to "3d Movie" and verifying
    # this false assertion as SUCCESSFUL. CPython gives "3D Movie", so
    # this must be a real FAILED.
    assert "3d movie".title() == "3d Movie"


main()
