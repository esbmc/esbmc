def main() -> None:
    # "hello" with e->i, l->p is "hippo", not "hello".
    assert "hello".translate(str.maketrans("el", "ip")) == "hello"


main()
