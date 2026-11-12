def ints() -> list[int]:
    return [1, 2]


def main() -> None:
    a: list[int] = ints()
    b: list[float] = [1.0, 2.0]
    assert a[0:] == b
    assert a[1:] != b
    c = [True, 2]
    assert c[:1] == [1]
    n = [[1], [2]]
    assert n[1:] == [[2]]
    assert (n + [[3]])[2] == [3]
    s = ["ab", "c"]
    assert s[1:] == ["c"]
    assert s + ["d"] != ["ab", "c", "e"]


main()
