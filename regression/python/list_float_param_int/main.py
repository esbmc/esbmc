import math


def first(p: list[float]) -> float:
    return p[0]


def main() -> None:
    # A list[float]-annotated parameter fed an int list misread its elements:
    # the constant-index read took the "pure float" fast path and reinterpreted
    # the int payloads as float garbage (~0). Reading now dispatches on the
    # stored element type_id, so the int payloads are promoted to float.
    assert first([3, 4]) == 3.0
    assert first([3.0, 4.0]) == 3.0  # genuine float list still works

    # math.dist (whose model takes list[float] parameters) with integer
    # coordinates was the visible symptom — it returned ~0.
    assert math.dist([0, 0], [3, 4]) == 5.0
    assert math.dist([0], [5]) == 5.0
    assert math.dist([0.0, 0.0], [3.0, 4.0]) == 5.0


main()
