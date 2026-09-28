# Call sites that disagree on the element type leave the parameter untyped
# rather than letting the first one win for the rest.


def size_of(xs: list) -> int:
    return len(xs)


def main() -> None:
    assert size_of([5, 3]) == 2
    assert size_of(["a", "b", "c"]) == 3


main()
