def main() -> None:
    # str.format() with a format spec ({:.2f}, {:>5}, ...) was rejected with
    # "format() missing keyword argument" — the field was treated as a keyword
    # because the ":spec" suffix was not split off. The spec is now applied to
    # the original constant value.
    assert "{:.2f}".format(3.14159) == "3.14"
    assert "{:5}".format("hi") == "hi   "
    assert "{:>5}".format("hi") == "   hi"
    assert "{:^6}".format("hi") == "  hi  "
    assert "{:05d}".format(42) == "00042"
    assert "{:x}".format(255) == "ff"
    assert "{:+d}".format(5) == "+5"
    assert "{:08.2f}".format(3.14) == "00003.14"
    assert "{:<05d}".format(7) == "70000"   # explicit align + 0 flag
    assert "{:>05d}".format(7) == "00007"

    # Indexed and keyword fields with specs.
    assert "{0:.2f}-{1:.1f}".format(3.14159, 2.5) == "3.14-2.5"
    assert "{x:.2f}".format(x=3.14159) == "3.14"

    # Plain replacement fields are unchanged.
    assert "{} {}".format(1, 2) == "1 2"
    assert "{1}{0}".format("a", "b") == "ba"


main()
