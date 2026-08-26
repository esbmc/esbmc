# An eager consumer of a generator dropped the generator's `if` clauses on the
# C++ side, so `sum(x for x in xs if x > 2)` silently summed every element and
# reported a wrong answer. The list-comprehension lowering keeps the filter.
def main() -> None:
    xs = [1, 2, 3, 4]

    assert sum(x for x in xs if x > 2) == 7
    assert min(x for x in xs if x > 2) == 3
    assert max(x for x in xs if x < 3) == 2
    assert len(sorted(x for x in xs if x > 2)) == 2

    # An unfiltered generator was already correct and must stay so.
    assert sum(x for x in xs) == 10

    # More than one condition, and a condition over a second name.
    assert sum(x for x in xs if x > 1 if x < 4) == 5

    # The list-comprehension spelling keeps working.
    assert sum([x for x in xs if x % 2 == 0]) == 6


main()
