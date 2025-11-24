def bar(y: float | None = None) -> None:
    if y is not None:
        assert y > 0.0

bar(-3.14)
