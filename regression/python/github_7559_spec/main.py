# A format spec pads the repr text, which is not modelled; the part is left
# unconstrained (#7559).
def f(s: str) -> None:
    t = f"{s!r:>6}"
    assert len(s) == 2


f("ab")
