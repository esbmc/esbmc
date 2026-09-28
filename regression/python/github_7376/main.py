def main() -> None:
    t = "hello"
    t = t + " world"
    r = t.replace("l", "L")
    assert r == "heLLo worLd"


main()
