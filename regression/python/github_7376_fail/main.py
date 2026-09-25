def main() -> None:
    t = "hello"
    t = t + " world"
    r = t.replace("l", "L")
    assert r == "hello world"


main()
