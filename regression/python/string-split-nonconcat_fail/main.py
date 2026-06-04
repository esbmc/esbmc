def main() -> None:
    # Unsoundness from issue #5096: split() of a join() receiver was modeled as
    # a length-1 list, so this FALSE assertion was wrongly proved SUCCESSFUL.
    # CPython yields len == 2, so ESBMC must report a counterexample.
    assert len(",".join(["a", "b"]).split(",")) == 1


main()
