def main() -> None:
    a = {1, 2}
    b = {3, 4}
    # a and b are disjoint, so the intersection is empty; 1 is not in it.
    i = a.intersection(b)
    assert 1 in i


main()
