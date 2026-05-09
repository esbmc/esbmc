def main() -> None:
    # issubset
    a = {1, 2}
    b = {1, 2, 3}
    assert a.issubset(b)
    assert not b.issubset(a)
    # update mutates self
    s = {1, 2}
    s.update({3})
    assert 3 in s and 1 in s
    # frozenset constructor
    fs = frozenset([1, 2])
    assert 1 in fs and 2 in fs
main()
