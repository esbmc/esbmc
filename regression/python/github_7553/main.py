# A dict keys view is set-like: it compares equal to a set with the same
# elements, in either order and regardless of ordering. A values view is not
# set-like, and a plain list still never equals a set (#7553).
def main() -> None:
    assert {1: 1}.keys() == {1}
    assert {1} == {1: 1}.keys()
    assert {1: 1, 2: 2}.keys() == {2, 1}
    assert {1: 1}.keys() != {2}
    assert {1: 1}.values() != {1}
    assert [1] != {1}
    assert {1: 1}.keys() != [1]
    assert len({1: 1}.keys()) == 1


main()
