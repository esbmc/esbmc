def main() -> None:
    # max()/min() over multiple non-numeric arguments (here two strings) is
    # rejected with a clean diagnostic instead of crashing the SMT backend in
    # compute_pointer_offset. The single-iterable form max([...]) is supported.
    assert max("apple", "banana") == "banana"


main()
