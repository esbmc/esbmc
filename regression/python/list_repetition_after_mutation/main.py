# `xs * n` expanded the declaring literal's elements at convert time, so a list
# mutated since its literal repeated the stale contents: [1, 2, 3] appended to
# and doubled produced six elements where Python gives eight.
def main() -> None:
    a = [1, 2, 3]
    a.append(4)
    assert len(a * 2) == 8

    b = [1, 2, 3]
    b.extend([4])
    assert len(b * 2) == 8

    c = [1, 2, 3]
    c.insert(0, 4)
    assert len(c * 2) == 8

    # An unmutated literal still repeats from the literal.
    d = [1, 2, 3]
    assert len(d * 2) == 6

    # The repeated contents are the mutated ones, not just the length.
    e = [1, 2]
    e.append(3)
    rep = e * 2
    assert rep[2] == 3
    assert rep[5] == 3


main()
