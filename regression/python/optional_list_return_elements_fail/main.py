def first_positive(values):
    for i in range(len(values)):
        if values[i] > 0:
            return list(values)
    return None


def main() -> None:
    r = first_positive([1, 1])
    assert r is not None
    seen = [False] * 2
    for x in r:
        assert not seen[x]
        seen[x] = True


main()
