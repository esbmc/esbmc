def main() -> None:
    # Digits are uncased, so a letter after a digit starts a new word
    # (CPython: cased-ness, not alnum-ness, drives word boundaries).
    assert "3d movie".title() == "3D Movie"
    assert "a1a".title() == "A1A"
    assert "x2y3z".title() == "X2Y3Z"
    assert "123abc".title() == "123Abc"
    # Unchanged behaviour around non-digit boundaries.
    assert "hello world".title() == "Hello World"
    assert "HELLO WORLD".title() == "Hello World"
    assert "it's a dog".title() == "It'S A Dog"
    assert "abc-def".title() == "Abc-Def"
    assert "mIxEd CaSe".title() == "Mixed Case"
    assert "item 2 title".title() == "Item 2 Title"
    assert "".title() == ""
    assert "123".title() == "123"


main()
