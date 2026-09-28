def main() -> None:
    # {1} and {2} share no element, so isdisjoint() is True and `not` makes the
    # assertion false. A method on a container literal does not resolve to a
    # class, and the unresolved-method fallback used to be a null void* — which
    # reads as False, so this was *proved* instead of left unknown.
    assert not {1}.isdisjoint({2})


main()
