# __ESBMC_dict_eq compared int and float dict values by raw type_id + bytes,
# so Python's numeric equality (1 == 1.0) was rejected as a spurious mismatch
# whenever one dict's values were floats and the other's were ints -- the same
# class of bug already fixed for __ESBMC_list_eq (#5207). Mentioned in #5444
# ("int/float __ESBMC_dict_eq" gap) as one of the blockers for
# quixbugs/shortest_paths, whose result dict mixes a float('inf') sentinel
# with int-valued min() results, compared against an int-literal `expected`.
# Covers both homogeneous dicts (all-float vs all-int) and a heterogeneous
# dict (mixed float/int values in one dict), which take different storage
# paths (dict_construction.cpp's all_values_float gate).
def main() -> None:
    d1 = {"a": 1.0, "b": 2.0}
    d2 = {"a": 1, "b": 2}
    assert d1 == d2
    assert not (d1 != d2)

    # Heterogeneous dict (mixed float/int values in the same dict), matching
    # shortest_paths' actual pattern: a float('inf') sentinel alongside
    # int-valued min() results, compared against an int-literal dict.
    d3 = {"a": float("inf"), "b": 2}
    d4 = {"a": 999999, "b": 2}
    assert d3 != d4

    d5 = {"a": 1, "b": 2.0}
    d6 = {"a": 1, "b": 2}
    assert d5 == d6


main()
