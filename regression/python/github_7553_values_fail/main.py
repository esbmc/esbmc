# A values view is deliberately not set-like: it compares unequal to a set
# whatever it holds. Marking it set-like alongside keys() would prove this
# one, which CPython says is false (#7553).
def main() -> None:
    assert {1: 1}.values() == {1}


main()
