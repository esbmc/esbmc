# The same inline call-result comparison, for shapes whose elements are
# modelled: these must keep verifying after the materialisation change (#7555).
def main() -> None:
    assert list([1, 2]) == [1, 2]
    assert list(range(3)) == [0, 1, 2]


main()
