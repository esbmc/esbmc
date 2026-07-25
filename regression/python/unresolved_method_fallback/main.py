def main() -> None:
    # A receiver that does resolve keeps its exact set semantics — the nondet
    # fallback for an unresolved method must not leak into this path.
    a = {1, 2}
    b = {3}
    assert a.isdisjoint(b)
    c = {2}
    assert not a.isdisjoint(c)
    assert a.issuperset(c)
    assert c.issubset(a)


main()
