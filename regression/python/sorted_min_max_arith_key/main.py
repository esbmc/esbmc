def main() -> None:
    # Arithmetic key lambda: previously silently dropped, yielding false verdicts.
    assert sorted([1, 2, 3], key=lambda v: -v) == [3, 2, 1]
    assert sorted([5, 1, 4, 2, 3], key=lambda v: v % 3) == [3, 1, 4, 5, 2]

    # Identity and unary keys over a constant list.
    assert sorted([3, 1, 2], key=lambda v: v) == [1, 2, 3]

    # list.sort(key=) rewrites to sorted(); the arithmetic key now applies.
    a = [1, 2, 3]
    a.sort(key=lambda v: -v)
    assert a == [3, 2, 1]

    # min/max with an arithmetic key select the right element.
    assert max([1, -5, 3], key=lambda v: -v) == -5
    assert min([1, -5, 3], key=lambda v: v * v) == 1

    # Existing folded forms still work: abs/len builtins and a tuple subscript.
    assert max([1, -5, 3], key=abs) == -5
    assert sorted(["aaa", "b", "cc"], key=len) == ["b", "cc", "aaa"]
    pairs = sorted([(0, 5), (1, 3), (2, 1)], key=lambda p: p[1])
    assert pairs == [(2, 1), (1, 3), (0, 5)]


main()
