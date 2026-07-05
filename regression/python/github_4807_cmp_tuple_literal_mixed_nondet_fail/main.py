# Negative companion to github_4807_cmp_tuple_literal_mixed_nondet: a
# mixed list-vs-tuple-literal comparison that the assert-unpack
# preprocessor pass does not rewrite (`!=`) lowers to an unconstrained
# nondet bool, so asserting it must fail. It must NOT constant-fold to
# True via the cross-category Eq/NotEq fold: tuple(<list>) is modelled as
# a list object, so the pair may really be tuple-vs-tuple. Must be
# updated when mixed comparison is properly lowered.


if __name__ == "__main__":
    assert tuple([1, 2]) != (1, 2)  # nondet bool -- may be false
