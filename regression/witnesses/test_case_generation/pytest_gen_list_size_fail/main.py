def short(x: list[int]) -> None:
    assert len(x) < 2


x = nondet_list(2)
short(x)
