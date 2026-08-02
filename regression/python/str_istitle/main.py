def main() -> None:
    # Titlecased strings: uppercase only after uncased chars, lowercase only
    # after cased ones, at least one cased char.
    assert "Hello World".istitle()
    assert "A".istitle()
    assert "It'S A Dog".istitle()
    assert "Hello  World".istitle()
    assert "Hello-World".istitle()
    assert "3D Movie".istitle()
    # Not titlecased.
    assert not "Hello world".istitle()
    assert not "".istitle()
    assert not "HELLO".istitle()
    assert not "aA".istitle()
    assert not "123".istitle()
    assert not "It's A Dog".istitle()
    assert not "3d Movie".istitle()


main()
