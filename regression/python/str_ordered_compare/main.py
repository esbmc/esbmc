def main() -> None:
    # Ordered comparison of constant strings was folded as `!equal`, so <= / >=
    # were wrong whenever the answer differed from inequality. All six relational
    # operators must agree with CPython's lexicographic ordering.
    assert "abc" <= "abc"          # equal -> True (was False)
    assert "abc" <= "abd"          # less -> True
    assert not ("abd" <= "abc")    # greater -> False (was True)

    assert "abc" >= "abc"          # equal -> True (was False)
    assert "abd" >= "abc"          # greater -> True
    assert not ("abc" >= "abd")    # less -> False

    assert not ("abc" < "abc")     # strict, equal -> False
    assert "abc" < "abd"
    assert "abd" > "abc"
    assert "ab" < "abc"            # shorter prefix orders first
    assert "ab" <= "ab"            # equal prefix
    assert "Z" < "a"               # uppercase sorts before lowercase (ASCII)

    # Equality is unchanged.
    assert "abc" == "abc"
    assert "abc" != "abd"


main()
