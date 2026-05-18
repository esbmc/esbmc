def outer(x: int) -> None:
    cap: int = 5

    def closure() -> None:
        # y has no explicit annotation -> annotator must infer the type
        # of `cap` via the closure RHS lookup (the patched path).
        y = cap
        assert y == 99

    closure()


outer(0)
