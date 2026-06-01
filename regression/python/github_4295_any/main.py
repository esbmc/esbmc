def main() -> None:
    t = (False, False, True)
    assert any(t)
    assert any((False, True))
    assert not any((False, False))

main()
