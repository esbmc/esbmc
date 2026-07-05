def f(c: bool) -> None:
    # x is int here. A str reassignment INSIDE the conditional must NOT retype
    # x: that would be unsound at the join. With correct gating the str write is
    # dropped and x stays int(7); if conditional retyping leaked, later reads of
    # x would wrongly see a str and the assertion below would fail.
    x: int = 7
    if c:
        x = "leaked"
    assert x == 7


f(False)
