def main() -> None:
    # Variable index on a homogeneous tuple.
    t = (10, 20, 30)
    i = 1
    assert t[i] == 20
    j = 0
    assert t[j] == 10
    k = 2
    assert t[k] == 30

    # Negative variable index (Python style: -1 is last).
    n = -1
    assert t[n] == 30

    # for-loop over a tuple desugars to t[i].
    s = 0
    for x in t:
        s += x
    assert s == 60

    # Tuple of strings: same homogeneous-element rule.
    names = ("a", "b", "c")
    idx = 1
    assert names[idx] == "b"
main()
