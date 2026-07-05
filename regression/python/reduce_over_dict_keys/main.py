def main() -> None:
    # max()/min()/sum() over a dict iterate its KEYS in CPython. The bare
    # builtins reinterpreted the dict struct as a list and gave wrong results;
    # they now route through d.keys() like list(d)/sorted(d) already did.
    d = {3: 1, 1: 2}
    assert max(d) == 3
    assert min(d) == 1
    assert sum(d) == 4

    # Unaffected: the variadic max/min form (not a single dict iterable).
    assert max(3, 7) == 7


main()
