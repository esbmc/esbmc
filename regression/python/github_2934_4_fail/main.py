def fun(x: int | str | None) -> None:
    if x is not None:
        assert x == 42

fun("hi")
