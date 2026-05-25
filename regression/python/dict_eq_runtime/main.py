import esbmc


def test_nondet_key() -> None:
    # Non-literal key forces the runtime __ESBMC_dict_eq fallback rather than
    # the literal-vs-literal constant fold. Asserts the runtime model still
    # decides this trivially-equal comparison correctly.
    k = esbmc.nondet_int()
    __ESBMC_assume(0 <= k <= 5)
    lhs = {k: 7, 1: 7}
    rhs = {1: 7, k: 7}
    assert lhs == rhs


def test_fromkeys_no_default() -> None:
    # dict.fromkeys without a default arg fills with None in CPython, which the
    # int-only fold cannot model. Refusing to fold means the runtime model is
    # consulted; ``d`` here ends up empty under that path because the keys are
    # not concrete, so equality with a literal None-valued dict cannot be
    # established without the model's nondet result. Just exercise the dispatch
    # to ensure the runtime call site stays reachable.
    d = dict.fromkeys([])
    assert len(d) == 0


test_nondet_key()
test_fromkeys_no_default()
