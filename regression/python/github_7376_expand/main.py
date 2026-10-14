def main() -> None:
    t = "abc"
    t = t + "abc"
    r = t.replace("a", "XY")
    assert r == "XYbcXYbc"


main()
