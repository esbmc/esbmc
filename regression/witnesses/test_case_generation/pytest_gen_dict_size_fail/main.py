def empty(d: dict[int, int]) -> None:
    assert len(d) < 1


d = nondet_dict(1, key_type=nondet_int(), value_type=nondet_int())
empty(d)
