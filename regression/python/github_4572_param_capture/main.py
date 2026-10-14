def outer(cap: int) -> None:
    # Closure captures `cap`, which is a *parameter* of `outer`.
    # `y = cap` has no explicit annotation, so the annotator must infer
    # the type by looking up `cap` in the enclosing function's args list.
    def closure() -> None:
        y = cap
        assert y == 0

    closure()


outer(0)
