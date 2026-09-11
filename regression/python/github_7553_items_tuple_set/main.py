# An items view is modelled by a placeholder holding the dict's keys, so a
# comparison against a set of (key, value) tuples answered from contents that
# are not the view's -- proving this inequality, which CPython makes false.
def main() -> None:
    assert {1: 1}.items() != {(1, 1)}


main()
