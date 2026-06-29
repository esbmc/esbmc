def main() -> None:
    # "%(x)s" % {"x": "hi"} is "hi", not "bye".
    assert "%(x)s" % {"x": "hi"} == "bye"


main()
